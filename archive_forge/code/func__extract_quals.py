from . import model
from .commontypes import COMMON_TYPES, resolve_common_type
from .error import FFIError, CDefError
import weakref, re, sys
def _extract_quals(self, type):
    quals = 0
    if isinstance(type, (pycparser.c_ast.TypeDecl, pycparser.c_ast.PtrDecl)):
        if 'const' in type.quals:
            quals |= model.Q_CONST
        if 'volatile' in type.quals:
            quals |= model.Q_VOLATILE
        if 'restrict' in type.quals:
            quals |= model.Q_RESTRICT
    return quals