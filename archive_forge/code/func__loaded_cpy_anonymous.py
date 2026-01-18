import sys
from . import model
from .error import VerificationError
from . import _imp_emulation as imp
def _loaded_cpy_anonymous(self, tp, name, module, **kwds):
    if isinstance(tp, model.EnumType):
        self._loaded_cpy_enum(tp, name, module, **kwds)
    else:
        self._loaded_struct_or_union(tp)