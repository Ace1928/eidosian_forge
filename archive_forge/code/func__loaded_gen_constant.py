import sys, os
import types
from . import model
from .error import VerificationError
def _loaded_gen_constant(self, tp, name, module, library):
    is_int = isinstance(tp, model.PrimitiveType) and tp.is_integer_type()
    value = self._load_constant(is_int, tp, name, module)
    setattr(library, name, value)
    type(library)._cffi_dir.append(name)