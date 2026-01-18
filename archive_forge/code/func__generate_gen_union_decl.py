import sys, os
import types
from . import model
from .error import VerificationError
def _generate_gen_union_decl(self, tp, name):
    assert name == tp.name
    self._generate_struct_or_union_decl(tp, 'union', name)