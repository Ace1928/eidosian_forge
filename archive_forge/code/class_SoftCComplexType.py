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
class SoftCComplexType(CComplexType):
    """
    a**b in Python can return either a complex or a float
    depending on the sign of a. This "soft complex" type is
    stored as a C complex (and so is a little slower than a
    direct C double) but it prints/coerces to a float if
    the imaginary part is 0. Therefore it provides a C
    representation of the Python behaviour.
    """
    to_py_function = '__pyx_Py_FromSoftComplex'

    def __init__(self):
        super(SoftCComplexType, self).__init__(c_double_type)

    def declaration_code(self, entity_code, for_display=0, dll_linkage=None, pyrex=0):
        base_result = super(SoftCComplexType, self).declaration_code(entity_code, for_display=for_display, dll_linkage=dll_linkage, pyrex=pyrex)
        if for_display:
            return 'soft %s' % base_result
        else:
            return base_result

    def create_to_py_utility_code(self, env):
        env.use_utility_code(UtilityCode.load_cached('SoftComplexToPy', 'Complex.c'))
        return True

    def __repr__(self):
        result = super(SoftCComplexType, self).__repr__()
        assert result[-1] == '>'
        return '%s (soft)%s' % (result[:-1], result[-1])