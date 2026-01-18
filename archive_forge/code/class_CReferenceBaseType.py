from __future__ import absolute_import
import copy
import hashlib
import re
from functools import partial
from itertools import product
from Cython.Utils import cached_function
from .Code import UtilityCode, LazyUtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from .Errors import error, CannotSpecialize, performance_hint
class CReferenceBaseType(BaseType):
    is_fake_reference = 0
    subtypes = ['ref_base_type']

    def __init__(self, base_type):
        self.ref_base_type = base_type

    def __repr__(self):
        return '<%r %s>' % (self.__class__.__name__, self.ref_base_type)

    def specialize(self, values):
        base_type = self.ref_base_type.specialize(values)
        if base_type == self.ref_base_type:
            return self
        else:
            return type(self)(base_type)

    def deduce_template_params(self, actual):
        return self.ref_base_type.deduce_template_params(actual)

    def __getattr__(self, name):
        return getattr(self.ref_base_type, name)