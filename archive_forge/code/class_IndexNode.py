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
class IndexNode(_IndexingBaseNode):
    subexprs = ['base', 'index']
    type_indices = None
    is_subscript = True
    is_fused_index = False

    def calculate_constant_result(self):
        self.constant_result = self.base.constant_result[self.index.constant_result]

    def compile_time_value(self, denv):
        base = self.base.compile_time_value(denv)
        index = self.index.compile_time_value(denv)
        try:
            return base[index]
        except Exception as e:
            self.compile_time_value_error(e)

    def is_simple(self):
        base = self.base
        return base.is_simple() and self.index.is_simple() and base.type and (base.type.is_ptr or base.type.is_array)

    def may_be_none(self):
        base_type = self.base.type
        if base_type:
            if base_type.is_string:
                return False
            if isinstance(self.index, SliceNode):
                if base_type in (bytes_type, bytearray_type, str_type, unicode_type, basestring_type, list_type, tuple_type):
                    return False
        return ExprNode.may_be_none(self)

    def analyse_target_declaration(self, env):
        pass

    def analyse_as_type(self, env):
        base_type = self.base.analyse_as_type(env)
        if base_type:
            if base_type.is_cpp_class or base_type.python_type_constructor_name:
                if self.index.is_sequence_constructor:
                    template_values = self.index.args
                else:
                    template_values = [self.index]
                type_node = Nodes.TemplatedTypeNode(pos=self.pos, positional_args=template_values, keyword_args=None)
                return type_node.analyse(env, base_type=base_type)
            elif self.index.is_slice or self.index.is_sequence_constructor:
                from . import MemoryView
                env.use_utility_code(MemoryView.view_utility_code)
                axes = [self.index] if self.index.is_slice else list(self.index.args)
                return PyrexTypes.MemoryViewSliceType(base_type, MemoryView.get_axes_specs(env, axes))
            elif not base_type.is_pyobject:
                index = self.index.compile_time_value(env)
                if index is not None:
                    try:
                        index = int(index)
                    except (ValueError, TypeError):
                        pass
                    else:
                        return PyrexTypes.CArrayType(base_type, index)
                error(self.pos, 'Array size must be a compile time constant')
        return None

    def analyse_pytyping_modifiers(self, env):
        modifiers = []
        modifier_node = self
        while modifier_node.is_subscript:
            modifier_type = modifier_node.base.analyse_as_type(env)
            if modifier_type and modifier_type.python_type_constructor_name and modifier_type.modifier_name:
                modifiers.append(modifier_type.modifier_name)
            modifier_node = modifier_node.index
        return modifiers

    def type_dependencies(self, env):
        return self.base.type_dependencies(env) + self.index.type_dependencies(env)

    def infer_type(self, env):
        base_type = self.base.infer_type(env)
        if self.index.is_slice:
            if base_type.is_string:
                return bytes_type
            elif base_type.is_pyunicode_ptr:
                return unicode_type
            elif base_type in (unicode_type, bytes_type, str_type, bytearray_type, list_type, tuple_type):
                return base_type
            elif base_type.is_memoryviewslice:
                return base_type
            else:
                return py_object_type
        index_type = self.index.infer_type(env)
        if index_type and index_type.is_int or isinstance(self.index, IntNode):
            if base_type is unicode_type:
                return PyrexTypes.c_py_ucs4_type
            elif base_type is str_type:
                return base_type
            elif base_type is bytearray_type:
                return PyrexTypes.c_uchar_type
            elif isinstance(self.base, BytesNode):
                return py_object_type
            elif base_type in (tuple_type, list_type):
                item_type = infer_sequence_item_type(env, self.base, self.index, seq_type=base_type)
                if item_type is not None:
                    return item_type
            elif base_type.is_ptr or base_type.is_array:
                return base_type.base_type
            elif base_type.is_ctuple and isinstance(self.index, IntNode):
                if self.index.has_constant_result():
                    index = self.index.constant_result
                    if index < 0:
                        index += base_type.size
                    if 0 <= index < base_type.size:
                        return base_type.components[index]
            elif base_type.is_memoryviewslice:
                if base_type.ndim == 0:
                    pass
                if base_type.ndim == 1:
                    return base_type.dtype
                else:
                    return PyrexTypes.MemoryViewSliceType(base_type.dtype, base_type.axes[1:])
        if self.index.is_sequence_constructor and base_type.is_memoryviewslice:
            inferred_type = base_type
            for a in self.index.args:
                if not inferred_type.is_memoryviewslice:
                    break
                inferred_type = IndexNode(self.pos, base=ExprNode(self.base.pos, type=inferred_type), index=a).infer_type(env)
            else:
                return inferred_type
        if base_type.is_cpp_class:

            class FakeOperand:

                def __init__(self, **kwds):
                    self.__dict__.update(kwds)
            operands = [FakeOperand(pos=self.pos, type=base_type), FakeOperand(pos=self.pos, type=index_type)]
            index_func = env.lookup_operator('[]', operands)
            if index_func is not None:
                return index_func.type.return_type
        if is_pythran_expr(base_type) and is_pythran_expr(index_type):
            index_with_type = (self.index, index_type)
            return PythranExpr(pythran_indexing_type(base_type, [index_with_type]))
        if base_type in (unicode_type, str_type):
            return base_type
        else:
            return py_object_type

    def analyse_types(self, env):
        return self.analyse_base_and_index_types(env, getting=True)

    def analyse_target_types(self, env):
        node = self.analyse_base_and_index_types(env, setting=True)
        if node.type.is_const:
            error(self.pos, 'Assignment to const dereference')
        if node is self and (not node.is_lvalue()):
            error(self.pos, "Assignment to non-lvalue of type '%s'" % node.type)
        return node

    def analyse_base_and_index_types(self, env, getting=False, setting=False, analyse_base=True):
        if analyse_base:
            self.base = self.base.analyse_types(env)
        if self.base.type.is_error:
            self.type = PyrexTypes.error_type
            return self
        is_slice = self.index.is_slice
        if not env.directives['wraparound']:
            if is_slice:
                check_negative_indices(self.index.start, self.index.stop)
            else:
                check_negative_indices(self.index)
        if not is_slice and isinstance(self.index, IntNode) and Utils.long_literal(self.index.value):
            self.index = self.index.coerce_to_pyobject(env)
        is_memslice = self.base.type.is_memoryviewslice
        if not is_memslice and (isinstance(self.base, BytesNode) or is_slice):
            if self.base.type.is_string or not (self.base.type.is_ptr or self.base.type.is_array):
                self.base = self.base.coerce_to_pyobject(env)
        replacement_node = self.analyse_as_buffer_operation(env, getting)
        if replacement_node is not None:
            return replacement_node
        self.nogil = env.nogil
        base_type = self.base.type
        if not base_type.is_cfunction:
            self.index = self.index.analyse_types(env)
            self.original_index_type = self.index.type
            if self.original_index_type.is_reference:
                self.original_index_type = self.original_index_type.ref_base_type
            if base_type.is_unicode_char:
                if setting:
                    warning(self.pos, 'cannot assign to Unicode string index', level=1)
                elif self.index.constant_result in (0, -1):
                    return self.base
                self.base = self.base.coerce_to_pyobject(env)
                base_type = self.base.type
        if base_type.is_pyobject:
            return self.analyse_as_pyobject(env, is_slice, getting, setting)
        elif base_type.is_ptr or base_type.is_array:
            return self.analyse_as_c_array(env, is_slice)
        elif base_type.is_cpp_class:
            return self.analyse_as_cpp(env, setting)
        elif base_type.is_cfunction:
            return self.analyse_as_c_function(env)
        elif base_type.is_ctuple:
            return self.analyse_as_c_tuple(env, getting, setting)
        else:
            error(self.pos, "Attempting to index non-array type '%s'" % base_type)
            self.type = PyrexTypes.error_type
            return self

    def analyse_as_pyobject(self, env, is_slice, getting, setting):
        base_type = self.base.type
        if self.index.type.is_unicode_char and base_type is not dict_type:
            warning(self.pos, 'Item lookup of unicode character codes now always converts to a Unicode string. Use an explicit C integer cast to get back the previous integer lookup behaviour.', level=1)
            self.index = self.index.coerce_to_pyobject(env)
            self.is_temp = 1
        elif self.index.type.is_int and base_type is not dict_type:
            if getting and (not env.directives['boundscheck']) and (base_type in (list_type, tuple_type, bytearray_type)) and (not self.index.type.signed or not env.directives['wraparound'] or (isinstance(self.index, IntNode) and self.index.has_constant_result() and (self.index.constant_result >= 0))):
                self.is_temp = 0
            else:
                self.is_temp = 1
            self.index = self.index.coerce_to(PyrexTypes.c_py_ssize_t_type, env).coerce_to_simple(env)
            self.original_index_type.create_to_py_utility_code(env)
        else:
            self.index = self.index.coerce_to_pyobject(env)
            self.is_temp = 1
        if self.index.type.is_int and base_type is unicode_type:
            self.type = PyrexTypes.c_py_ucs4_type
        elif self.index.type.is_int and base_type is bytearray_type:
            if setting:
                self.type = PyrexTypes.c_uchar_type
            else:
                self.type = PyrexTypes.c_int_type
        elif is_slice and base_type in (bytes_type, bytearray_type, str_type, unicode_type, list_type, tuple_type):
            self.type = base_type
        else:
            item_type = None
            if base_type in (list_type, tuple_type) and self.index.type.is_int:
                item_type = infer_sequence_item_type(env, self.base, self.index, seq_type=base_type)
            if base_type in (list_type, tuple_type, dict_type):
                self.base = self.base.as_none_safe_node("'NoneType' object is not subscriptable")
            if item_type is None or not item_type.is_pyobject:
                self.type = py_object_type
            else:
                self.type = item_type
        self.wrap_in_nonecheck_node(env, getting)
        return self

    def analyse_as_c_array(self, env, is_slice):
        base_type = self.base.type
        self.type = base_type.base_type
        if self.type.is_cpp_class:
            self.type = PyrexTypes.CReferenceType(self.type)
        if is_slice:
            self.type = base_type
        elif self.index.type.is_pyobject:
            self.index = self.index.coerce_to(PyrexTypes.c_py_ssize_t_type, env)
        elif not self.index.type.is_int:
            error(self.pos, "Invalid index type '%s'" % self.index.type)
        return self

    def analyse_as_cpp(self, env, setting):
        base_type = self.base.type
        function = env.lookup_operator('[]', [self.base, self.index])
        if function is None:
            error(self.pos, "Indexing '%s' not supported for index type '%s'" % (base_type, self.index.type))
            self.type = PyrexTypes.error_type
            self.result_code = '<error>'
            return self
        func_type = function.type
        if func_type.is_ptr:
            func_type = func_type.base_type
        self.exception_check = func_type.exception_check
        self.exception_value = func_type.exception_value
        if self.exception_check:
            if not setting:
                self.is_temp = True
            if needs_cpp_exception_conversion(self):
                env.use_utility_code(UtilityCode.load_cached('CppExceptionConversion', 'CppSupport.cpp'))
        self.index = self.index.coerce_to(func_type.args[0].type, env)
        self.type = func_type.return_type
        if setting and (not func_type.return_type.is_reference):
            error(self.pos, "Can't set non-reference result '%s'" % self.type)
        return self

    def analyse_as_c_function(self, env):
        base_type = self.base.type
        if base_type.is_fused:
            self.parse_indexed_fused_cdef(env)
        else:
            self.type_indices = self.parse_index_as_types(env)
            self.index = None
            if base_type.templates is None:
                error(self.pos, 'Can only parameterize template functions.')
                self.type = error_type
            elif self.type_indices is None:
                self.type = error_type
            elif len(base_type.templates) != len(self.type_indices):
                error(self.pos, 'Wrong number of template arguments: expected %s, got %s' % (len(base_type.templates), len(self.type_indices)))
                self.type = error_type
            else:
                self.type = base_type.specialize(dict(zip(base_type.templates, self.type_indices)))
        return self

    def analyse_as_c_tuple(self, env, getting, setting):
        base_type = self.base.type
        if isinstance(self.index, IntNode) and self.index.has_constant_result():
            index = self.index.constant_result
            if -base_type.size <= index < base_type.size:
                if index < 0:
                    index += base_type.size
                self.type = base_type.components[index]
            else:
                error(self.pos, "Index %s out of bounds for '%s'" % (index, base_type))
                self.type = PyrexTypes.error_type
            return self
        else:
            self.base = self.base.coerce_to_pyobject(env)
            return self.analyse_base_and_index_types(env, getting=getting, setting=setting, analyse_base=False)

    def analyse_as_buffer_operation(self, env, getting):
        """
        Analyse buffer indexing and memoryview indexing/slicing
        """
        if isinstance(self.index, TupleNode):
            indices = self.index.args
        else:
            indices = [self.index]
        base = self.base
        base_type = base.type
        replacement_node = None
        if base_type.is_memoryviewslice:
            from . import MemoryView
            if base.is_memview_slice:
                merged_indices = base.merged_indices(indices)
                if merged_indices is not None:
                    base = base.base
                    base_type = base.type
                    indices = merged_indices
            have_slices, indices, newaxes = MemoryView.unellipsify(indices, base_type.ndim)
            if have_slices:
                replacement_node = MemoryViewSliceNode(self.pos, indices=indices, base=base)
            else:
                replacement_node = MemoryViewIndexNode(self.pos, indices=indices, base=base)
        elif base_type.is_buffer or base_type.is_pythran_expr:
            if base_type.is_pythran_expr or len(indices) == base_type.ndim:
                is_buffer_access = True
                indices = [index.analyse_types(env) for index in indices]
                if base_type.is_pythran_expr:
                    do_replacement = all((index.type.is_int or index.is_slice or index.type.is_pythran_expr for index in indices))
                    if do_replacement:
                        for i, index in enumerate(indices):
                            if index.is_slice:
                                index = SliceIntNode(index.pos, start=index.start, stop=index.stop, step=index.step)
                                index = index.analyse_types(env)
                                indices[i] = index
                else:
                    do_replacement = all((index.type.is_int for index in indices))
                if do_replacement:
                    replacement_node = BufferIndexNode(self.pos, indices=indices, base=base)
                    assert not isinstance(self.index, CloneNode)
        if replacement_node is not None:
            replacement_node = replacement_node.analyse_types(env, getting)
        return replacement_node

    def wrap_in_nonecheck_node(self, env, getting):
        if not env.directives['nonecheck'] or not self.base.may_be_none():
            return
        self.base = self.base.as_none_safe_node("'NoneType' object is not subscriptable")

    def parse_index_as_types(self, env, required=True):
        if isinstance(self.index, TupleNode):
            indices = self.index.args
        else:
            indices = [self.index]
        type_indices = []
        for index in indices:
            type_indices.append(index.analyse_as_type(env))
            if type_indices[-1] is None:
                if required:
                    error(index.pos, 'not parsable as a type')
                return None
        return type_indices

    def parse_indexed_fused_cdef(self, env):
        """
        Interpret fused_cdef_func[specific_type1, ...]

        Note that if this method is called, we are an indexed cdef function
        with fused argument types, and this IndexNode will be replaced by the
        NameNode with specific entry just after analysis of expressions by
        AnalyseExpressionsTransform.
        """
        self.type = PyrexTypes.error_type
        self.is_fused_index = True
        base_type = self.base.type
        positions = []
        if self.index.is_name or self.index.is_attribute:
            positions.append(self.index.pos)
        elif isinstance(self.index, TupleNode):
            for arg in self.index.args:
                positions.append(arg.pos)
        specific_types = self.parse_index_as_types(env, required=False)
        if specific_types is None:
            self.index = self.index.analyse_types(env)
            if not self.base.entry.as_variable:
                error(self.pos, 'Can only index fused functions with types')
            else:
                self.base.entry = self.entry = self.base.entry.as_variable
                self.base.type = self.type = self.entry.type
                self.base.is_temp = True
                self.is_temp = True
                self.entry.used = True
            self.is_fused_index = False
            return
        for i, type in enumerate(specific_types):
            specific_types[i] = type.specialize_fused(env)
        fused_types = base_type.get_fused_types()
        if len(specific_types) > len(fused_types):
            return error(self.pos, 'Too many types specified')
        elif len(specific_types) < len(fused_types):
            t = fused_types[len(specific_types)]
            return error(self.pos, 'Not enough types specified to specialize the function, %s is still fused' % t)
        for pos, specific_type, fused_type in zip(positions, specific_types, fused_types):
            if not any([specific_type.same_as(t) for t in fused_type.types]):
                return error(pos, 'Type not in fused type')
            if specific_type is None or specific_type.is_error:
                return
        fused_to_specific = dict(zip(fused_types, specific_types))
        type = base_type.specialize(fused_to_specific)
        if type.is_fused:
            error(self.pos, 'Index operation makes function only partially specific')
        else:
            for signature in self.base.type.get_all_specialized_function_types():
                if type.same_as(signature):
                    self.type = signature
                    if self.base.is_attribute:
                        self.entry = signature.entry
                        self.is_attribute = True
                        self.obj = self.base.obj
                    self.type.entry.used = True
                    self.base.type = signature
                    self.base.entry = signature.entry
                    break
            else:
                raise InternalError("Couldn't find the right signature")
    gil_message = 'Indexing Python object'

    def calculate_result_code(self):
        if self.base.type in (list_type, tuple_type, bytearray_type):
            if self.base.type is list_type:
                index_code = 'PyList_GET_ITEM(%s, %s)'
            elif self.base.type is tuple_type:
                index_code = 'PyTuple_GET_ITEM(%s, %s)'
            elif self.base.type is bytearray_type:
                index_code = '((unsigned char)(PyByteArray_AS_STRING(%s)[%s]))'
            else:
                assert False, 'unexpected base type in indexing: %s' % self.base.type
        elif self.base.type.is_cfunction:
            return '%s<%s>' % (self.base.result(), ','.join([param.empty_declaration_code() for param in self.type_indices]))
        elif self.base.type.is_ctuple:
            index = self.index.constant_result
            if index < 0:
                index += self.base.type.size
            return '%s.f%s' % (self.base.result(), index)
        else:
            if (self.type.is_ptr or self.type.is_array) and self.type == self.base.type:
                error(self.pos, 'Invalid use of pointer slice')
                return
            index_code = '(%s[%s])'
        return index_code % (self.base.result(), self.index.result())

    def extra_index_params(self, code):
        if self.index.type.is_int:
            is_list = self.base.type is list_type
            wraparound = bool(code.globalstate.directives['wraparound']) and self.original_index_type.signed and (not (isinstance(self.index.constant_result, _py_int_types) and self.index.constant_result >= 0))
            boundscheck = bool(code.globalstate.directives['boundscheck'])
            return ', %s, %d, %s, %d, %d, %d' % (self.original_index_type.empty_declaration_code(), self.original_index_type.signed and 1 or 0, self.original_index_type.to_py_function, is_list, wraparound, boundscheck)
        else:
            return ''

    def generate_result_code(self, code):
        if not self.is_temp:
            return
        utility_code = None
        error_value = None
        if self.type.is_pyobject:
            error_value = 'NULL'
            if self.index.type.is_int:
                if self.base.type is list_type:
                    function = '__Pyx_GetItemInt_List'
                elif self.base.type is tuple_type:
                    function = '__Pyx_GetItemInt_Tuple'
                else:
                    function = '__Pyx_GetItemInt'
                utility_code = TempitaUtilityCode.load_cached('GetItemInt', 'ObjectHandling.c')
            elif self.base.type is dict_type:
                function = '__Pyx_PyDict_GetItem'
                utility_code = UtilityCode.load_cached('DictGetItem', 'ObjectHandling.c')
            elif self.base.type is py_object_type and self.index.type in (str_type, unicode_type):
                function = '__Pyx_PyObject_Dict_GetItem'
                utility_code = UtilityCode.load_cached('DictGetItem', 'ObjectHandling.c')
            else:
                function = '__Pyx_PyObject_GetItem'
                code.globalstate.use_utility_code(TempitaUtilityCode.load_cached('GetItemInt', 'ObjectHandling.c'))
                utility_code = UtilityCode.load_cached('ObjectGetItem', 'ObjectHandling.c')
        elif self.type.is_unicode_char and self.base.type is unicode_type:
            assert self.index.type.is_int
            function = '__Pyx_GetItemInt_Unicode'
            error_value = '(Py_UCS4)-1'
            utility_code = UtilityCode.load_cached('GetItemIntUnicode', 'StringTools.c')
        elif self.base.type is bytearray_type:
            assert self.index.type.is_int
            assert self.type.is_int
            function = '__Pyx_GetItemInt_ByteArray'
            error_value = '-1'
            utility_code = UtilityCode.load_cached('GetItemIntByteArray', 'StringTools.c')
        elif not (self.base.type.is_cpp_class and self.exception_check):
            assert False, 'unexpected type %s and base type %s for indexing (%s)' % (self.type, self.base.type, self.pos)
        if utility_code is not None:
            code.globalstate.use_utility_code(utility_code)
        if self.index.type.is_int:
            index_code = self.index.result()
        else:
            index_code = self.index.py_result()
        if self.base.type.is_cpp_class and self.exception_check:
            translate_cpp_exception(code, self.pos, '%s = %s[%s];' % (self.result(), self.base.result(), self.index.result()), self.result() if self.type.is_pyobject else None, self.exception_value, self.in_nogil_context)
        else:
            error_check = '!%s' if error_value == 'NULL' else '%%s == %s' % error_value
            code.putln('%s = %s(%s, %s%s); %s' % (self.result(), function, self.base.py_result(), index_code, self.extra_index_params(code), code.error_goto_if(error_check % self.result(), self.pos)))
        if self.type.is_pyobject:
            self.generate_gotref(code)

    def generate_setitem_code(self, value_code, code):
        if self.index.type.is_int:
            if self.base.type is bytearray_type:
                code.globalstate.use_utility_code(UtilityCode.load_cached('SetItemIntByteArray', 'StringTools.c'))
                function = '__Pyx_SetItemInt_ByteArray'
            else:
                code.globalstate.use_utility_code(UtilityCode.load_cached('SetItemInt', 'ObjectHandling.c'))
                function = '__Pyx_SetItemInt'
            index_code = self.index.result()
        else:
            index_code = self.index.py_result()
            if self.base.type is dict_type:
                function = 'PyDict_SetItem'
            else:
                function = 'PyObject_SetItem'
        code.putln(code.error_goto_if_neg('%s(%s, %s, %s%s)' % (function, self.base.py_result(), index_code, value_code, self.extra_index_params(code)), self.pos))

    def generate_assignment_code(self, rhs, code, overloaded_assignment=False, exception_check=None, exception_value=None):
        self.generate_subexpr_evaluation_code(code)
        if self.type.is_pyobject:
            self.generate_setitem_code(rhs.py_result(), code)
        elif self.base.type is bytearray_type:
            value_code = self._check_byte_value(code, rhs)
            self.generate_setitem_code(value_code, code)
        elif self.base.type.is_cpp_class and self.exception_check and (self.exception_check == '+'):
            if overloaded_assignment and exception_check and (self.exception_value != exception_value):
                translate_double_cpp_exception(code, self.pos, self.type, self.result(), rhs.result(), self.exception_value, exception_value, self.in_nogil_context)
            else:
                translate_cpp_exception(code, self.pos, '%s = %s;' % (self.result(), rhs.result()), self.result() if self.type.is_pyobject else None, self.exception_value, self.in_nogil_context)
        else:
            code.putln('%s = %s;' % (self.result(), rhs.result()))
        self.generate_subexpr_disposal_code(code)
        self.free_subexpr_temps(code)
        rhs.generate_disposal_code(code)
        rhs.free_temps(code)

    def _check_byte_value(self, code, rhs):
        assert rhs.type.is_int, repr(rhs.type)
        value_code = rhs.result()
        if rhs.has_constant_result():
            if 0 <= rhs.constant_result < 256:
                return value_code
            needs_cast = True
            warning(rhs.pos, 'value outside of range(0, 256) when assigning to byte: %s' % rhs.constant_result, level=1)
        else:
            needs_cast = rhs.type != PyrexTypes.c_uchar_type
        if not self.nogil:
            conditions = []
            if rhs.is_literal or rhs.type.signed:
                conditions.append('%s < 0' % value_code)
            if rhs.is_literal or not (rhs.is_temp and rhs.type in (PyrexTypes.c_uchar_type, PyrexTypes.c_char_type, PyrexTypes.c_schar_type)):
                conditions.append('%s > 255' % value_code)
            if conditions:
                code.putln('if (unlikely(%s)) {' % ' || '.join(conditions))
                code.putln('PyErr_SetString(PyExc_ValueError, "byte must be in range(0, 256)"); %s' % code.error_goto(self.pos))
                code.putln('}')
        if needs_cast:
            value_code = '((unsigned char)%s)' % value_code
        return value_code

    def generate_deletion_code(self, code, ignore_nonexisting=False):
        self.generate_subexpr_evaluation_code(code)
        if self.index.type.is_int:
            function = '__Pyx_DelItemInt'
            index_code = self.index.result()
            code.globalstate.use_utility_code(UtilityCode.load_cached('DelItemInt', 'ObjectHandling.c'))
        else:
            index_code = self.index.py_result()
            if self.base.type is dict_type:
                function = 'PyDict_DelItem'
            else:
                function = 'PyObject_DelItem'
        code.putln(code.error_goto_if_neg('%s(%s, %s%s)' % (function, self.base.py_result(), index_code, self.extra_index_params(code)), self.pos))
        self.generate_subexpr_disposal_code(code)
        self.free_subexpr_temps(code)