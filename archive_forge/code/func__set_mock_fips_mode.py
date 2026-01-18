from __future__ import division
import hashlib
import logging; log = logging.getLogger(__name__)
import re
import os
from struct import Struct
from warnings import warn
from passlib import exc
from passlib.utils import join_bytes, to_native_str, join_byte_values, to_bytes, \
from passlib.utils.compat import irange, int_types, unicode_or_bytes_types, PY3, error_from
from passlib.utils.decor import memoized_property
def _set_mock_fips_mode(enable=True):
    """
    UT helper which monkeypatches lookup_hash() internals to replicate FIPS mode.
    """
    global mock_fips_mode
    mock_fips_mode = enable
    lookup_hash.clear_cache()