from . import (
from .Errors import error
from . import PyrexTypes
from .UtilityCode import CythonUtilityCode
from .Code import TempitaUtilityCode, UtilityCode
from .Visitor import PrintTree, TreeVisitor, VisitorTransform
class UFuncConversion(object):

    def __init__(self, node):
        self.node = node
        self.global_scope = node.local_scope.global_scope()
        self.in_definitions = self.get_in_type_info()
        self.out_definitions = self.get_out_type_info()

    def get_in_type_info(self):
        definitions = []
        for n, arg in enumerate(self.node.args):
            type_const = _get_type_constant(self.node.pos, arg.type)
            definitions.append(_ArgumentInfo(arg.type, type_const))
        return definitions

    def get_out_type_info(self):
        if self.node.return_type.is_ctuple:
            components = self.node.return_type.components
        else:
            components = [self.node.return_type]
        definitions = []
        for n, type in enumerate(components):
            definitions.append(_ArgumentInfo(type, _get_type_constant(self.node.pos, type)))
        return definitions

    def generate_cy_utility_code(self):
        arg_types = [a.type for a in self.in_definitions]
        out_types = [a.type for a in self.out_definitions]
        inline_func_decl = self.node.entry.type.declaration_code(self.node.entry.cname, pyrex=True)
        self.node.entry.used = True
        ufunc_cname = self.global_scope.next_id(self.node.entry.name + '_ufunc_def')
        will_be_called_without_gil = not (any((t.is_pyobject for t in arg_types)) or any((t.is_pyobject for t in out_types)))
        context = dict(func_cname=ufunc_cname, in_types=arg_types, out_types=out_types, inline_func_call=self.node.entry.cname, inline_func_declaration=inline_func_decl, nogil=self.node.entry.type.nogil, will_be_called_without_gil=will_be_called_without_gil)
        code = CythonUtilityCode.load('UFuncDefinition', 'UFuncs.pyx', context=context, outer_module_scope=self.global_scope)
        tree = code.get_tree(entries_only=True)
        return tree

    def use_generic_utility_code(self):
        self.global_scope.use_utility_code(UtilityCode.load_cached('UFuncsInit', 'UFuncs_C.c'))
        self.global_scope.use_utility_code(UtilityCode.load_cached('NumpyImportUFunc', 'NumpyImportArray.c'))