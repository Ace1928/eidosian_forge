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
def _load_overflow_base(env):
    env.use_utility_code(UtilityCode.load('Common', 'Overflow.c'))
    for type in ('int', 'long', 'long long'):
        env.use_utility_code(TempitaUtilityCode.load_cached('BaseCaseSigned', 'Overflow.c', context={'INT': type, 'NAME': type.replace(' ', '_')}))
    for type in ('unsigned int', 'unsigned long', 'unsigned long long'):
        env.use_utility_code(TempitaUtilityCode.load_cached('BaseCaseUnsigned', 'Overflow.c', context={'UINT': type, 'NAME': type.replace(' ', '_')}))