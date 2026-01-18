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
def _get_record_list(self, category=None):
    """return list of records for category (cached)

        this is an internal helper used only by identify_record()
        """
    try:
        return self._record_lists[category]
    except KeyError:
        pass
    value = self._record_lists[category] = [self.get_record(scheme, category) for scheme in self.schemes]
    return value