from __future__ import with_statement
import re
import logging; log = logging.getLogger(__name__)
import threading
import time
from warnings import warn
from passlib import exc
from passlib.exc import ExpectedStringError, ExpectedTypeError, PasslibConfigWarning
from passlib.registry import get_crypt_handler, _validate_handler_name
from passlib.utils import (handlers as uh, to_bytes,
from passlib.utils.binary import BASE64_CHARS
from passlib.utils.compat import (iteritems, num_types, irange,
from passlib.utils.decor import deprecated_method, memoized_property
def _get_record_options_with_flag(self, scheme, category):
    """return composite dict of options for given scheme + category.

        this is currently a private method, though some variant
        of its output may eventually be made public.

        given a scheme & category, it returns two things:
        a set of all the keyword options to pass to :meth:`_create_record`,
        and a bool flag indicating whether any of these options
        were specific to the named category. if this flag is false,
        the options are identical to the options for the default category.

        the options dict includes all the scheme-specific settings,
        as well as optional *deprecated* keyword.
        """
    kwds, has_cat_options = self.get_scheme_options_with_flag(scheme, category)
    value, not_inherited = self.is_deprecated_with_flag(scheme, category)
    if value:
        kwds['deprecated'] = True
    if not_inherited:
        has_cat_options = True
    return (kwds, has_cat_options)