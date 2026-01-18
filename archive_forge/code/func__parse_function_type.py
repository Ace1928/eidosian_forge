from . import model
from .commontypes import COMMON_TYPES, resolve_common_type
from .error import FFIError, CDefError
import weakref, re, sys
def _parse_function_type(self, typenode, funcname=None):
    params = list(getattr(typenode.args, 'params', []))
    for i, arg in enumerate(params):
        if not hasattr(arg, 'type'):
            raise CDefError("%s arg %d: unknown type '%s' (if you meant to use the old C syntax of giving untyped arguments, it is not supported)" % (funcname or 'in expression', i + 1, getattr(arg, 'name', '?')))
    ellipsis = len(params) > 0 and isinstance(params[-1].type, pycparser.c_ast.TypeDecl) and isinstance(params[-1].type.type, pycparser.c_ast.IdentifierType) and (params[-1].type.type.names == ['__dotdotdot__'])
    if ellipsis:
        params.pop()
        if not params:
            raise CDefError("%s: a function with only '(...)' as argument is not correct C" % (funcname or 'in expression'))
    args = [self._as_func_arg(*self._get_type_and_quals(argdeclnode.type)) for argdeclnode in params]
    if not ellipsis and args == [model.void_type]:
        args = []
    result, quals = self._get_type_and_quals(typenode.type)
    abi = None
    if hasattr(typenode.type, 'quals'):
        if typenode.type.quals[-3:] == ['volatile', 'volatile', 'const']:
            abi = '__stdcall'
    return model.RawFunctionType(tuple(args), result, ellipsis, abi)