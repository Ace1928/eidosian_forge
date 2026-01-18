import sys, os
import types
from . import model
from .error import VerificationError
def _generate_gen_macro_decl(self, tp, name):
    if tp == '...':
        check_value = None
    else:
        check_value = tp
    self._generate_gen_const(True, name, check_value=check_value)