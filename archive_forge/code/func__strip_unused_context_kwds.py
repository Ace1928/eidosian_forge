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
def _strip_unused_context_kwds(self, kwds, record):
    """
        helper which removes any context keywords from **kwds**
        that are known to be used by another scheme in this context,
        but are NOT supported by handler specified by **record**.

        .. note::
            as optimization, load() will set this method to None on a per-instance basis
            if there are no context kwds.
        """
    if not kwds:
        return
    unused_kwds = self._config.context_kwds.difference(record.context_kwds)
    for key in unused_kwds:
        kwds.pop(key, None)