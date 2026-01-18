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
def _write_to_parser(self, parser, section):
    """helper to write to ConfigParser instance"""
    render_key = self._render_config_key
    render_value = self._render_ini_value
    parser.add_section(section)
    for k, v in self._config.iter_config():
        v = render_value(k, v)
        k = render_key(k)
        parser.set(section, k, v)