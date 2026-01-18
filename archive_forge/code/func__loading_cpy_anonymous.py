import sys
from . import model
from .error import VerificationError
from . import _imp_emulation as imp
def _loading_cpy_anonymous(self, tp, name, module):
    if isinstance(tp, model.EnumType):
        self._loading_cpy_enum(tp, name, module)
    else:
        self._loading_struct_or_union(tp, '', name, module)