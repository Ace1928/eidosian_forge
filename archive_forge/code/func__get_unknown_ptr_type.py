from . import model
from .commontypes import COMMON_TYPES, resolve_common_type
from .error import FFIError, CDefError
import weakref, re, sys
def _get_unknown_ptr_type(self, decl):
    if decl.type.type.type.names == ['__dotdotdot__']:
        return model.unknown_ptr_type(decl.name)
    raise FFIError(':%d: unsupported usage of "..." in typedef' % decl.coord.line)