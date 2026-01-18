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
def _generate_dummy_refcounting_assignment(self, code, cname, rhs_cname, *ignored_args, **ignored_kwds):
    if self.needs_refcounting:
        raise NotImplementedError('Ref-counting operation not yet implemented for type %s' % self)
    code.putln('%s = %s' % (cname, rhs_cname))