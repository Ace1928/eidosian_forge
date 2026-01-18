from . import model
from .commontypes import COMMON_TYPES, resolve_common_type
from .error import FFIError, CDefError
import weakref, re, sys
def _make_partial(self, tp, nested):
    if not isinstance(tp, model.StructOrUnion):
        raise CDefError('%s cannot be partial' % (tp,))
    if not tp.has_c_name() and (not nested):
        raise NotImplementedError('%s is partial but has no C name' % (tp,))
    tp.partial = True