from . import model
from .commontypes import COMMON_TYPES, resolve_common_type
from .error import FFIError, CDefError
import weakref, re, sys
def _parse_decl(self, decl):
    node = decl.type
    if isinstance(node, pycparser.c_ast.FuncDecl):
        tp, quals = self._get_type_and_quals(node, name=decl.name)
        assert isinstance(tp, model.RawFunctionType)
        self._declare_function(tp, quals, decl)
    else:
        if isinstance(node, pycparser.c_ast.Struct):
            self._get_struct_union_enum_type('struct', node)
        elif isinstance(node, pycparser.c_ast.Union):
            self._get_struct_union_enum_type('union', node)
        elif isinstance(node, pycparser.c_ast.Enum):
            self._get_struct_union_enum_type('enum', node)
        elif not decl.name:
            raise CDefError('construct does not declare any variable', decl)
        if decl.name:
            tp, quals = self._get_type_and_quals(node, partial_length_ok=True)
            if tp.is_raw_function:
                self._declare_function(tp, quals, decl)
            elif tp.is_integer_type() and hasattr(decl, 'init') and hasattr(decl.init, 'value') and _r_int_literal.match(decl.init.value):
                self._add_integer_constant(decl.name, decl.init.value)
            elif tp.is_integer_type() and isinstance(decl.init, pycparser.c_ast.UnaryOp) and (decl.init.op == '-') and hasattr(decl.init.expr, 'value') and _r_int_literal.match(decl.init.expr.value):
                self._add_integer_constant(decl.name, '-' + decl.init.expr.value)
            elif tp is model.void_type and decl.name.startswith('__cffi_extern_python_'):
                self._inside_extern_python = decl.name
            else:
                if self._inside_extern_python != '__cffi_extern_python_stop':
                    raise CDefError('cannot declare constants or variables with \'extern "Python"\'')
                if quals & model.Q_CONST and (not tp.is_array_type):
                    self._declare('constant ' + decl.name, tp, quals=quals)
                else:
                    _warn_for_non_extern_non_static_global_variable(decl)
                    self._declare('variable ' + decl.name, tp, quals=quals)