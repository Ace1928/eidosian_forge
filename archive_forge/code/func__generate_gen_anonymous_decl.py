import sys, os
import types
from . import model
from .error import VerificationError
def _generate_gen_anonymous_decl(self, tp, name):
    if isinstance(tp, model.EnumType):
        self._generate_gen_enum_decl(tp, name, '')
    else:
        self._generate_struct_or_union_decl(tp, '', name)