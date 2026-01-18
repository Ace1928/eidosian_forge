from __future__ import absolute_import
import cython
import re
import sys
import copy
import os.path
import operator
from .Errors import (
from .Code import UtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from . import Nodes
from .Nodes import Node, utility_code_for_imports, SingleAssignmentNode
from . import PyrexTypes
from .PyrexTypes import py_object_type, typecast, error_type, \
from . import TypeSlots
from .Builtin import (
from . import Builtin
from . import Symtab
from .. import Utils
from .Annotate import AnnotationItem
from . import Future
from ..Debugging import print_call_chain
from .DebugFlags import debug_disposal_code, debug_coercion
from .Pythran import (to_pythran, is_pythran_supported_type, is_pythran_supported_operation_type,
from .PyrexTypes import PythranExpr
class SequenceNode(ExprNode):
    subexprs = ['args', 'mult_factor']
    is_sequence_constructor = 1
    unpacked_items = None
    mult_factor = None
    slow = False

    def compile_time_value_list(self, denv):
        return [arg.compile_time_value(denv) for arg in self.args]

    def replace_starred_target_node(self):
        self.starred_assignment = False
        args = []
        for arg in self.args:
            if arg.is_starred:
                if self.starred_assignment:
                    error(arg.pos, 'more than 1 starred expression in assignment')
                self.starred_assignment = True
                arg = arg.target
                arg.is_starred = True
            args.append(arg)
        self.args = args

    def analyse_target_declaration(self, env):
        self.replace_starred_target_node()
        for arg in self.args:
            arg.analyse_target_declaration(env)

    def analyse_types(self, env, skip_children=False):
        for i, arg in enumerate(self.args):
            if not skip_children:
                arg = arg.analyse_types(env)
            self.args[i] = arg.coerce_to_pyobject(env)
        if self.mult_factor:
            mult_factor = self.mult_factor.analyse_types(env)
            if not mult_factor.type.is_int:
                mult_factor = mult_factor.coerce_to_pyobject(env)
            self.mult_factor = mult_factor.coerce_to_simple(env)
        self.is_temp = 1
        return self

    def coerce_to_ctuple(self, dst_type, env):
        if self.type == dst_type:
            return self
        assert not self.mult_factor
        if len(self.args) != dst_type.size:
            error(self.pos, 'trying to coerce sequence to ctuple of wrong length, expected %d, got %d' % (dst_type.size, len(self.args)))
        coerced_args = [arg.coerce_to(type, env) for arg, type in zip(self.args, dst_type.components)]
        return TupleNode(self.pos, args=coerced_args, type=dst_type, is_temp=True)

    def _create_merge_node_if_necessary(self, env):
        self._flatten_starred_args()
        if not any((arg.is_starred for arg in self.args)):
            return self
        args = []
        values = []
        for arg in self.args:
            if arg.is_starred:
                if values:
                    args.append(TupleNode(values[0].pos, args=values).analyse_types(env, skip_children=True))
                    values = []
                args.append(arg.target)
            else:
                values.append(arg)
        if values:
            args.append(TupleNode(values[0].pos, args=values).analyse_types(env, skip_children=True))
        node = MergedSequenceNode(self.pos, args, self.type)
        if self.mult_factor:
            node = binop_node(self.pos, '*', node, self.mult_factor.coerce_to_pyobject(env), inplace=True, type=self.type, is_temp=True)
        return node

    def _flatten_starred_args(self):
        args = []
        for arg in self.args:
            if arg.is_starred and arg.target.is_sequence_constructor and (not arg.target.mult_factor):
                args.extend(arg.target.args)
            else:
                args.append(arg)
        self.args[:] = args

    def may_be_none(self):
        return False

    def analyse_target_types(self, env):
        if self.mult_factor:
            error(self.pos, "can't assign to multiplied sequence")
        self.unpacked_items = []
        self.coerced_unpacked_items = []
        self.any_coerced_items = False
        for i, arg in enumerate(self.args):
            arg = self.args[i] = arg.analyse_target_types(env)
            if arg.is_starred:
                if not arg.type.assignable_from(list_type):
                    error(arg.pos, 'starred target must have Python object (list) type')
                if arg.type is py_object_type:
                    arg.type = list_type
            unpacked_item = PyTempNode(self.pos, env)
            coerced_unpacked_item = unpacked_item.coerce_to(arg.type, env)
            if unpacked_item is not coerced_unpacked_item:
                self.any_coerced_items = True
            self.unpacked_items.append(unpacked_item)
            self.coerced_unpacked_items.append(coerced_unpacked_item)
        self.type = py_object_type
        return self

    def generate_result_code(self, code):
        self.generate_operation_code(code)

    def generate_sequence_packing_code(self, code, target=None, plain=False):
        if target is None:
            target = self.result()
        size_factor = c_mult = ''
        mult_factor = None
        if self.mult_factor and (not plain):
            mult_factor = self.mult_factor
            if mult_factor.type.is_int:
                c_mult = mult_factor.result()
                if isinstance(mult_factor.constant_result, _py_int_types) and mult_factor.constant_result > 0:
                    size_factor = ' * %s' % mult_factor.constant_result
                elif mult_factor.type.signed:
                    size_factor = ' * ((%s<0) ? 0:%s)' % (c_mult, c_mult)
                else:
                    size_factor = ' * (%s)' % (c_mult,)
        if self.type is tuple_type and (self.is_literal or self.slow) and (not c_mult):
            code.putln('%s = PyTuple_Pack(%d, %s); %s' % (target, len(self.args), ', '.join((arg.py_result() for arg in self.args)), code.error_goto_if_null(target, self.pos)))
            code.put_gotref(target, py_object_type)
        elif self.type.is_ctuple:
            for i, arg in enumerate(self.args):
                code.putln('%s.f%s = %s;' % (target, i, arg.result()))
        else:
            if self.type is list_type:
                create_func, set_item_func = ('PyList_New', '__Pyx_PyList_SET_ITEM')
            elif self.type is tuple_type:
                create_func, set_item_func = ('PyTuple_New', '__Pyx_PyTuple_SET_ITEM')
            else:
                raise InternalError('sequence packing for unexpected type %s' % self.type)
            arg_count = len(self.args)
            code.putln('%s = %s(%s%s); %s' % (target, create_func, arg_count, size_factor, code.error_goto_if_null(target, self.pos)))
            code.put_gotref(target, py_object_type)
            if c_mult:
                counter = Naming.quick_temp_cname
                code.putln('{ Py_ssize_t %s;' % counter)
                if arg_count == 1:
                    offset = counter
                else:
                    offset = '%s * %s' % (counter, arg_count)
                code.putln('for (%s=0; %s < %s; %s++) {' % (counter, counter, c_mult, counter))
            else:
                offset = ''
            for i in range(arg_count):
                arg = self.args[i]
                if c_mult or not arg.result_in_temp():
                    code.put_incref(arg.result(), arg.ctype())
                arg.generate_giveref(code)
                code.putln('if (%s(%s, %s, %s)) %s;' % (set_item_func, target, (offset and i) and '%s + %s' % (offset, i) or (offset or i), arg.py_result(), code.error_goto(self.pos)))
            if c_mult:
                code.putln('}')
                code.putln('}')
        if mult_factor is not None and mult_factor.type.is_pyobject:
            code.putln('{ PyObject* %s = PyNumber_InPlaceMultiply(%s, %s); %s' % (Naming.quick_temp_cname, target, mult_factor.py_result(), code.error_goto_if_null(Naming.quick_temp_cname, self.pos)))
            code.put_gotref(Naming.quick_temp_cname, py_object_type)
            code.put_decref(target, py_object_type)
            code.putln('%s = %s;' % (target, Naming.quick_temp_cname))
            code.putln('}')

    def generate_subexpr_disposal_code(self, code):
        if self.mult_factor and self.mult_factor.type.is_int:
            super(SequenceNode, self).generate_subexpr_disposal_code(code)
        elif self.type is tuple_type and (self.is_literal or self.slow):
            super(SequenceNode, self).generate_subexpr_disposal_code(code)
        else:
            for arg in self.args:
                arg.generate_post_assignment_code(code)
            if self.mult_factor:
                self.mult_factor.generate_disposal_code(code)

    def generate_assignment_code(self, rhs, code, overloaded_assignment=False, exception_check=None, exception_value=None):
        if self.starred_assignment:
            self.generate_starred_assignment_code(rhs, code)
        else:
            self.generate_parallel_assignment_code(rhs, code)
        for item in self.unpacked_items:
            item.release(code)
        rhs.free_temps(code)
    _func_iternext_type = PyrexTypes.CPtrType(PyrexTypes.CFuncType(PyrexTypes.py_object_type, [PyrexTypes.CFuncTypeArg('it', PyrexTypes.py_object_type, None)]))

    def generate_parallel_assignment_code(self, rhs, code):
        for item in self.unpacked_items:
            item.allocate(code)
        special_unpack = rhs.type is py_object_type or rhs.type in (tuple_type, list_type) or (not rhs.type.is_builtin_type)
        long_enough_for_a_loop = len(self.unpacked_items) > 3
        if special_unpack:
            self.generate_special_parallel_unpacking_code(code, rhs, use_loop=long_enough_for_a_loop)
        else:
            code.putln('{')
            self.generate_generic_parallel_unpacking_code(code, rhs, self.unpacked_items, use_loop=long_enough_for_a_loop)
            code.putln('}')
        for value_node in self.coerced_unpacked_items:
            value_node.generate_evaluation_code(code)
        for i in range(len(self.args)):
            self.args[i].generate_assignment_code(self.coerced_unpacked_items[i], code)

    def generate_special_parallel_unpacking_code(self, code, rhs, use_loop):
        sequence_type_test = '1'
        none_check = 'likely(%s != Py_None)' % rhs.py_result()
        if rhs.type is list_type:
            sequence_types = ['List']
            if rhs.may_be_none():
                sequence_type_test = none_check
        elif rhs.type is tuple_type:
            sequence_types = ['Tuple']
            if rhs.may_be_none():
                sequence_type_test = none_check
        else:
            sequence_types = ['Tuple', 'List']
            tuple_check = 'likely(PyTuple_CheckExact(%s))' % rhs.py_result()
            list_check = 'PyList_CheckExact(%s)' % rhs.py_result()
            sequence_type_test = '(%s) || (%s)' % (tuple_check, list_check)
        code.putln('if (%s) {' % sequence_type_test)
        code.putln('PyObject* sequence = %s;' % rhs.py_result())
        code.putln('Py_ssize_t size = __Pyx_PySequence_SIZE(sequence);')
        code.putln('if (unlikely(size != %d)) {' % len(self.args))
        code.globalstate.use_utility_code(UtilityCode.load_cached('RaiseTooManyValuesToUnpack', 'ObjectHandling.c'))
        code.putln('if (size > %d) __Pyx_RaiseTooManyValuesError(%d);' % (len(self.args), len(self.args)))
        code.globalstate.use_utility_code(UtilityCode.load_cached('RaiseNeedMoreValuesToUnpack', 'ObjectHandling.c'))
        code.putln('else if (size >= 0) __Pyx_RaiseNeedMoreValuesError(size);')
        code.putln(code.error_goto(self.pos))
        code.putln('}')
        code.putln('#if CYTHON_ASSUME_SAFE_MACROS && !CYTHON_AVOID_BORROWED_REFS')
        if len(sequence_types) == 2:
            code.putln('if (likely(Py%s_CheckExact(sequence))) {' % sequence_types[0])
        for i, item in enumerate(self.unpacked_items):
            code.putln('%s = Py%s_GET_ITEM(sequence, %d); ' % (item.result(), sequence_types[0], i))
        if len(sequence_types) == 2:
            code.putln('} else {')
            for i, item in enumerate(self.unpacked_items):
                code.putln('%s = Py%s_GET_ITEM(sequence, %d); ' % (item.result(), sequence_types[1], i))
            code.putln('}')
        for item in self.unpacked_items:
            code.put_incref(item.result(), item.ctype())
        code.putln('#else')
        if not use_loop:
            for i, item in enumerate(self.unpacked_items):
                code.putln('%s = PySequence_ITEM(sequence, %d); %s' % (item.result(), i, code.error_goto_if_null(item.result(), self.pos)))
                code.put_gotref(item.result(), item.type)
        else:
            code.putln('{')
            code.putln('Py_ssize_t i;')
            code.putln('PyObject** temps[%s] = {%s};' % (len(self.unpacked_items), ','.join(['&%s' % item.result() for item in self.unpacked_items])))
            code.putln('for (i=0; i < %s; i++) {' % len(self.unpacked_items))
            code.putln('PyObject* item = PySequence_ITEM(sequence, i); %s' % code.error_goto_if_null('item', self.pos))
            code.put_gotref('item', py_object_type)
            code.putln('*(temps[i]) = item;')
            code.putln('}')
            code.putln('}')
        code.putln('#endif')
        rhs.generate_disposal_code(code)
        if sequence_type_test == '1':
            code.putln('}')
        elif sequence_type_test == none_check:
            code.putln('} else {')
            code.globalstate.use_utility_code(UtilityCode.load_cached('RaiseNoneIterError', 'ObjectHandling.c'))
            code.putln('__Pyx_RaiseNoneNotIterableError(); %s' % code.error_goto(self.pos))
            code.putln('}')
        else:
            code.putln('} else {')
            self.generate_generic_parallel_unpacking_code(code, rhs, self.unpacked_items, use_loop=use_loop)
            code.putln('}')

    def generate_generic_parallel_unpacking_code(self, code, rhs, unpacked_items, use_loop, terminate=True):
        code.globalstate.use_utility_code(UtilityCode.load_cached('RaiseNeedMoreValuesToUnpack', 'ObjectHandling.c'))
        code.globalstate.use_utility_code(UtilityCode.load_cached('IterFinish', 'ObjectHandling.c'))
        code.putln('Py_ssize_t index = -1;')
        if use_loop:
            code.putln('PyObject** temps[%s] = {%s};' % (len(self.unpacked_items), ','.join(['&%s' % item.result() for item in unpacked_items])))
        iterator_temp = code.funcstate.allocate_temp(py_object_type, manage_ref=True)
        code.putln('%s = PyObject_GetIter(%s); %s' % (iterator_temp, rhs.py_result(), code.error_goto_if_null(iterator_temp, self.pos)))
        code.put_gotref(iterator_temp, py_object_type)
        rhs.generate_disposal_code(code)
        iternext_func = code.funcstate.allocate_temp(self._func_iternext_type, manage_ref=False)
        code.putln('%s = __Pyx_PyObject_GetIterNextFunc(%s);' % (iternext_func, iterator_temp))
        unpacking_error_label = code.new_label('unpacking_failed')
        unpack_code = '%s(%s)' % (iternext_func, iterator_temp)
        if use_loop:
            code.putln('for (index=0; index < %s; index++) {' % len(unpacked_items))
            code.put('PyObject* item = %s; if (unlikely(!item)) ' % unpack_code)
            code.put_goto(unpacking_error_label)
            code.put_gotref('item', py_object_type)
            code.putln('*(temps[index]) = item;')
            code.putln('}')
        else:
            for i, item in enumerate(unpacked_items):
                code.put('index = %d; %s = %s; if (unlikely(!%s)) ' % (i, item.result(), unpack_code, item.result()))
                code.put_goto(unpacking_error_label)
                item.generate_gotref(code)
        if terminate:
            code.globalstate.use_utility_code(UtilityCode.load_cached('UnpackItemEndCheck', 'ObjectHandling.c'))
            code.put_error_if_neg(self.pos, '__Pyx_IternextUnpackEndCheck(%s, %d)' % (unpack_code, len(unpacked_items)))
            code.putln('%s = NULL;' % iternext_func)
            code.put_decref_clear(iterator_temp, py_object_type)
        unpacking_done_label = code.new_label('unpacking_done')
        code.put_goto(unpacking_done_label)
        code.put_label(unpacking_error_label)
        code.put_decref_clear(iterator_temp, py_object_type)
        code.putln('%s = NULL;' % iternext_func)
        code.putln('if (__Pyx_IterFinish() == 0) __Pyx_RaiseNeedMoreValuesError(index);')
        code.putln(code.error_goto(self.pos))
        code.put_label(unpacking_done_label)
        code.funcstate.release_temp(iternext_func)
        if terminate:
            code.funcstate.release_temp(iterator_temp)
            iterator_temp = None
        return iterator_temp

    def generate_starred_assignment_code(self, rhs, code):
        for i, arg in enumerate(self.args):
            if arg.is_starred:
                starred_target = self.unpacked_items[i]
                unpacked_fixed_items_left = self.unpacked_items[:i]
                unpacked_fixed_items_right = self.unpacked_items[i + 1:]
                break
        else:
            assert False
        iterator_temp = None
        if unpacked_fixed_items_left:
            for item in unpacked_fixed_items_left:
                item.allocate(code)
            code.putln('{')
            iterator_temp = self.generate_generic_parallel_unpacking_code(code, rhs, unpacked_fixed_items_left, use_loop=True, terminate=False)
            for i, item in enumerate(unpacked_fixed_items_left):
                value_node = self.coerced_unpacked_items[i]
                value_node.generate_evaluation_code(code)
            code.putln('}')
        starred_target.allocate(code)
        target_list = starred_target.result()
        code.putln('%s = %s(%s); %s' % (target_list, '__Pyx_PySequence_ListKeepNew' if not iterator_temp and rhs.is_temp and (rhs.type in (py_object_type, list_type)) else 'PySequence_List', iterator_temp or rhs.py_result(), code.error_goto_if_null(target_list, self.pos)))
        starred_target.generate_gotref(code)
        if iterator_temp:
            code.put_decref_clear(iterator_temp, py_object_type)
            code.funcstate.release_temp(iterator_temp)
        else:
            rhs.generate_disposal_code(code)
        if unpacked_fixed_items_right:
            code.globalstate.use_utility_code(UtilityCode.load_cached('RaiseNeedMoreValuesToUnpack', 'ObjectHandling.c'))
            length_temp = code.funcstate.allocate_temp(PyrexTypes.c_py_ssize_t_type, manage_ref=False)
            code.putln('%s = PyList_GET_SIZE(%s);' % (length_temp, target_list))
            code.putln('if (unlikely(%s < %d)) {' % (length_temp, len(unpacked_fixed_items_right)))
            code.putln('__Pyx_RaiseNeedMoreValuesError(%d+%s); %s' % (len(unpacked_fixed_items_left), length_temp, code.error_goto(self.pos)))
            code.putln('}')
            for item in unpacked_fixed_items_right[::-1]:
                item.allocate(code)
            for i, (item, coerced_arg) in enumerate(zip(unpacked_fixed_items_right[::-1], self.coerced_unpacked_items[::-1])):
                code.putln('#if CYTHON_COMPILING_IN_CPYTHON')
                code.putln('%s = PyList_GET_ITEM(%s, %s-%d); ' % (item.py_result(), target_list, length_temp, i + 1))
                code.putln('((PyVarObject*)%s)->ob_size--;' % target_list)
                code.putln('#else')
                code.putln('%s = PySequence_ITEM(%s, %s-%d); ' % (item.py_result(), target_list, length_temp, i + 1))
                code.putln('#endif')
                item.generate_gotref(code)
                coerced_arg.generate_evaluation_code(code)
            code.putln('#if !CYTHON_COMPILING_IN_CPYTHON')
            sublist_temp = code.funcstate.allocate_temp(py_object_type, manage_ref=True)
            code.putln('%s = PySequence_GetSlice(%s, 0, %s-%d); %s' % (sublist_temp, target_list, length_temp, len(unpacked_fixed_items_right), code.error_goto_if_null(sublist_temp, self.pos)))
            code.put_gotref(sublist_temp, py_object_type)
            code.funcstate.release_temp(length_temp)
            code.put_decref(target_list, py_object_type)
            code.putln('%s = %s; %s = NULL;' % (target_list, sublist_temp, sublist_temp))
            code.putln('#else')
            code.putln('CYTHON_UNUSED_VAR(%s);' % sublist_temp)
            code.funcstate.release_temp(sublist_temp)
            code.putln('#endif')
        for i, arg in enumerate(self.args):
            arg.generate_assignment_code(self.coerced_unpacked_items[i], code)

    def annotate(self, code):
        for arg in self.args:
            arg.annotate(code)
        if self.unpacked_items:
            for arg in self.unpacked_items:
                arg.annotate(code)
            for arg in self.coerced_unpacked_items:
                arg.annotate(code)