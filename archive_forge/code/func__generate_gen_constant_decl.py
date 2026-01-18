import sys, os
import types
from . import model
from .error import VerificationError
def _generate_gen_constant_decl(self, tp, name):
    is_int = isinstance(tp, model.PrimitiveType) and tp.is_integer_type()
    self._generate_gen_const(is_int, name, tp)