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
def _get_passlib_hasher(self, passlib_name):
    """
        resolve passlib hasher by name, using context if available.
        """
    context = self.context
    if context is None:
        return registry.get_crypt_handler(passlib_name)
    else:
        return context.handler(passlib_name)