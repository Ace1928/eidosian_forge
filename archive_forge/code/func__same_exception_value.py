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
def _same_exception_value(self, other_exc_value):
    if self.exception_value == other_exc_value:
        return 1
    if self.exception_check != '+':
        return 0
    if not self.exception_value or not other_exc_value:
        return 0
    if self.exception_value.type != other_exc_value.type:
        return 0
    if self.exception_value.entry and other_exc_value.entry:
        if self.exception_value.entry.cname != other_exc_value.entry.cname:
            return 0
    if self.exception_value.name != other_exc_value.name:
        return 0
    return 1