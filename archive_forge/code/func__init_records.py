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
def _init_records(self):
    self._record_lists = {}
    records = self._records = {}
    all_context_kwds = self.context_kwds = set()
    get_options = self._get_record_options_with_flag
    categories = (None,) + self.categories
    for handler in self.handlers:
        scheme = handler.name
        all_context_kwds.update(handler.context_kwds)
        for cat in categories:
            kwds, has_cat_options = get_options(scheme, cat)
            if cat is None or has_cat_options:
                records[scheme, cat] = self._create_record(handler, cat, **kwds)