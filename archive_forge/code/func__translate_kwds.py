from functools import update_wrapper, wraps
import logging; log = logging.getLogger(__name__)
import sys
import weakref
from warnings import warn
from passlib import exc, registry
from passlib.context import CryptContext
from passlib.exc import PasslibRuntimeWarning
from passlib.utils.compat import get_method_function, iteritems, OrderedDict, unicode
from passlib.utils.decor import memoized_property
@memoized_property
def _translate_kwds(self):
    """
        internal helper for safe_summary() --
        used to translate passlib hash options -> django keywords
        """
    out = dict(checksum='hash')
    if self._has_rounds and 'pbkdf2' in self.passlib_handler.name:
        out['rounds'] = 'iterations'
    return out