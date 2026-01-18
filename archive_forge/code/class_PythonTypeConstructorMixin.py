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
class PythonTypeConstructorMixin(object):
    """Used to help Cython interpret indexed types from the typing module (or similar)
    """
    modifier_name = None

    def set_python_type_constructor_name(self, name):
        self.python_type_constructor_name = name

    def specialize_here(self, pos, env, template_values=None):
        return self

    def __repr__(self):
        if self.base_type:
            return '%s[%r]' % (self.name, self.base_type)
        else:
            return self.name

    def is_template_type(self):
        return True