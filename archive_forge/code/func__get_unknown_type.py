from . import model
from .commontypes import COMMON_TYPES, resolve_common_type
from .error import FFIError, CDefError
import weakref, re, sys
def _get_unknown_type(self, decl):
    typenames = decl.type.type.names
    if typenames == ['__dotdotdot__']:
        return model.unknown_type(decl.name)
    if typenames == ['__dotdotdotint__']:
        if self._uses_new_feature is None:
            self._uses_new_feature = "'typedef int... %s'" % decl.name
        return model.UnknownIntegerType(decl.name)
    if typenames == ['__dotdotdotfloat__']:
        if self._uses_new_feature is None:
            self._uses_new_feature = "'typedef float... %s'" % decl.name
        return model.UnknownFloatType(decl.name)
    raise FFIError(':%d: unsupported usage of "..." in typedef' % decl.coord.line)