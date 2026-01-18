import sys, os
import types
from . import model
from .error import VerificationError
def _loading_gen_anonymous(self, tp, name, module):
    if isinstance(tp, model.EnumType):
        self._loading_gen_enum(tp, name, module, '')
    else:
        self._loading_struct_or_union(tp, '', name, module)