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
def _generate_decref(self, code, cname, nanny, null_check=False, clear=False, clear_before_decref=False):
    prefix = '__Pyx' if nanny else 'Py'
    X = 'X' if null_check else ''
    if nanny:
        code.funcstate.needs_refnanny = True
    if clear:
        if clear_before_decref:
            if not nanny:
                X = ''
            code.putln('%s_%sCLEAR(%s);' % (prefix, X, cname))
        else:
            code.putln('%s_%sDECREF(%s); %s = 0;' % (prefix, X, self.as_pyobject(cname), cname))
    else:
        code.putln('%s_%sDECREF(%s);' % (prefix, X, self.as_pyobject(cname)))