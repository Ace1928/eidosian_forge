import sys, os
import types
from . import model
from .error import VerificationError
def _loaded_gen_enum(self, tp, name, module, library):
    for enumerator, enumvalue in zip(tp.enumerators, tp.enumvalues):
        setattr(library, enumerator, enumvalue)
        type(library)._cffi_dir.append(enumerator)