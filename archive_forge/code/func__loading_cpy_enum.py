import sys
from . import model
from .error import VerificationError
from . import _imp_emulation as imp
def _loading_cpy_enum(self, tp, name, module):
    if tp.partial:
        enumvalues = [getattr(module, enumerator) for enumerator in tp.enumerators]
        tp.enumvalues = tuple(enumvalues)
        tp.partial_resolved = True