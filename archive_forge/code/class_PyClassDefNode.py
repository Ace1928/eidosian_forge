from __future__ import absolute_import
import cython
import sys, copy
from itertools import chain
from . import Builtin
from .Errors import error, warning, InternalError, CompileError, CannotSpecialize
from . import Naming
from . import PyrexTypes
from . import TypeSlots
from .PyrexTypes import py_object_type, error_type
from .Symtab import (ModuleScope, LocalScope, ClosureScope, PropertyScope,
from .Code import UtilityCode
from .StringEncoding import EncodedString
from . import Future
from . import Options
from . import DebugFlags
from .Pythran import has_np_pythran, pythran_type, is_pythran_buffer
from ..Utils import add_metaclass, str_to_number
class PyClassDefNode(ClassDefNode):
    child_attrs = ['doc_node', 'body', 'dict', 'metaclass', 'mkw', 'bases', 'class_result', 'target', 'class_cell', 'decorators', 'orig_bases']
    decorators = None
    class_result = None
    is_py3_style_class = False
    metaclass = None
    mkw = None
    doc_node = None
    orig_bases = None

    def __init__(self, pos, name, bases, doc, body, decorators=None, keyword_args=None, force_py3_semantics=False):
        StatNode.__init__(self, pos)
        self.name = name
        self.doc = doc
        self.body = body
        self.decorators = decorators
        self.bases = bases
        from . import ExprNodes
        if self.doc and Options.docstrings:
            doc = embed_position(self.pos, self.doc)
            doc_node = ExprNodes.StringNode(pos, value=doc)
            self.doc_node = ExprNodes.NameNode(name=EncodedString('__doc__'), type=py_object_type, pos=pos)
        else:
            doc_node = None
        allow_py2_metaclass = not force_py3_semantics
        if keyword_args:
            allow_py2_metaclass = False
            self.is_py3_style_class = True
            if keyword_args.is_dict_literal:
                if keyword_args.key_value_pairs:
                    for i, item in list(enumerate(keyword_args.key_value_pairs))[::-1]:
                        if item.key.value == 'metaclass':
                            if self.metaclass is not None:
                                error(item.pos, "keyword argument 'metaclass' passed multiple times")
                            self.metaclass = item.value
                            del keyword_args.key_value_pairs[i]
                    self.mkw = keyword_args
                else:
                    assert self.metaclass is not None
            else:
                self.mkw = ExprNodes.ProxyNode(keyword_args)
        if force_py3_semantics or self.bases or self.mkw or self.metaclass:
            if self.metaclass is None:
                if keyword_args and (not keyword_args.is_dict_literal):
                    mkdict = self.mkw
                else:
                    mkdict = None
                if not mkdict and self.bases.is_sequence_constructor and (not self.bases.args):
                    pass
                else:
                    self.metaclass = ExprNodes.PyClassMetaclassNode(pos, class_def_node=self)
                needs_metaclass_calculation = False
            else:
                needs_metaclass_calculation = True
            self.dict = ExprNodes.PyClassNamespaceNode(pos, name=name, doc=doc_node, class_def_node=self)
            self.classobj = ExprNodes.Py3ClassNode(pos, name=name, class_def_node=self, doc=doc_node, calculate_metaclass=needs_metaclass_calculation, allow_py2_metaclass=allow_py2_metaclass, force_type=force_py3_semantics)
        else:
            self.dict = ExprNodes.DictNode(pos, key_value_pairs=[])
            self.classobj = ExprNodes.ClassNode(pos, name=name, class_def_node=self, doc=doc_node)
        self.target = ExprNodes.NameNode(pos, name=name)
        self.class_cell = ExprNodes.ClassCellInjectorNode(self.pos)

    def as_cclass(self):
        """
        Return this node as if it were declared as an extension class
        """
        if self.is_py3_style_class:
            error(self.classobj.pos, 'Python3 style class could not be represented as C class')
            return
        from . import ExprNodes
        return CClassDefNode(self.pos, visibility='private', module_name=None, class_name=self.name, bases=self.bases or ExprNodes.TupleNode(self.pos, args=[]), decorators=self.decorators, body=self.body, in_pxd=False, doc=self.doc)

    def create_scope(self, env):
        genv = env
        while genv.is_py_class_scope or genv.is_c_class_scope:
            genv = genv.outer_scope
        cenv = self.scope = PyClassScope(name=self.name, outer_scope=genv)
        return cenv

    def analyse_declarations(self, env):
        unwrapped_class_result = class_result = self.classobj
        if self.decorators:
            from .ExprNodes import SimpleCallNode
            for decorator in self.decorators[::-1]:
                class_result = SimpleCallNode(decorator.pos, function=decorator.decorator, args=[class_result])
            self.decorators = None
        self.class_result = class_result
        if self.bases:
            self.bases.analyse_declarations(env)
        if self.mkw:
            self.mkw.analyse_declarations(env)
        self.class_result.analyse_declarations(env)
        self.target.analyse_target_declaration(env)
        cenv = self.create_scope(env)
        cenv.directives = env.directives
        cenv.class_obj_cname = self.target.entry.cname
        if self.doc_node:
            self.doc_node.analyse_target_declaration(cenv)
        self.body.analyse_declarations(cenv)
        unwrapped_class_result.analyse_annotations(cenv)
    update_bases_functype = PyrexTypes.CFuncType(PyrexTypes.py_object_type, [PyrexTypes.CFuncTypeArg('bases', PyrexTypes.py_object_type, None)])

    def analyse_expressions(self, env):
        if self.bases and (not (self.bases.is_sequence_constructor and len(self.bases.args) == 0)):
            from .ExprNodes import PythonCapiCallNode, CloneNode
            orig_bases = self.bases.analyse_expressions(env)
            self.bases = PythonCapiCallNode(orig_bases.pos, function_name='__Pyx_PEP560_update_bases', func_type=self.update_bases_functype, utility_code=UtilityCode.load_cached('Py3UpdateBases', 'ObjectHandling.c'), args=[CloneNode(orig_bases)])
            self.orig_bases = orig_bases
        if self.bases:
            self.bases = self.bases.analyse_expressions(env)
        if self.mkw:
            self.mkw = self.mkw.analyse_expressions(env)
        if self.metaclass:
            self.metaclass = self.metaclass.analyse_expressions(env)
        self.dict = self.dict.analyse_expressions(env)
        self.class_result = self.class_result.analyse_expressions(env)
        cenv = self.scope
        self.body = self.body.analyse_expressions(cenv)
        self.target = self.target.analyse_target_expression(env, self.classobj)
        self.class_cell = self.class_cell.analyse_expressions(cenv)
        return self

    def generate_function_definitions(self, env, code):
        self.generate_lambda_definitions(self.scope, code)
        self.body.generate_function_definitions(self.scope, code)

    def generate_execution_code(self, code):
        code.mark_pos(self.pos)
        code.pyclass_stack.append(self)
        cenv = self.scope
        if self.orig_bases:
            self.orig_bases.generate_evaluation_code(code)
        if self.bases:
            self.bases.generate_evaluation_code(code)
        if self.mkw:
            self.mkw.generate_evaluation_code(code)
        if self.metaclass:
            self.metaclass.generate_evaluation_code(code)
        self.dict.generate_evaluation_code(code)
        if self.orig_bases:
            code.putln('if (%s != %s) {' % (self.bases.result(), self.orig_bases.result()))
            code.putln(code.error_goto_if_neg('PyDict_SetItemString(%s, "__orig_bases__", %s)' % (self.dict.result(), self.orig_bases.result()), self.pos))
            code.putln('}')
            self.orig_bases.generate_disposal_code(code)
            self.orig_bases.free_temps(code)
        cenv.namespace_cname = cenv.class_obj_cname = self.dict.result()
        class_cell = self.class_cell
        if class_cell is not None and (not class_cell.is_active):
            class_cell = None
        if class_cell is not None:
            class_cell.generate_evaluation_code(code)
        self.body.generate_execution_code(code)
        self.class_result.generate_evaluation_code(code)
        if class_cell is not None:
            class_cell.generate_injection_code(code, self.class_result.result())
        if class_cell is not None:
            class_cell.generate_disposal_code(code)
            class_cell.free_temps(code)
        cenv.namespace_cname = cenv.class_obj_cname = self.classobj.result()
        self.target.generate_assignment_code(self.class_result, code)
        self.dict.generate_disposal_code(code)
        self.dict.free_temps(code)
        if self.metaclass:
            self.metaclass.generate_disposal_code(code)
            self.metaclass.free_temps(code)
        if self.mkw:
            self.mkw.generate_disposal_code(code)
            self.mkw.free_temps(code)
        if self.bases:
            self.bases.generate_disposal_code(code)
            self.bases.free_temps(code)
        code.pyclass_stack.pop()