from __future__ import annotations
import re
import sys
import warnings
from typing import (
from urllib.parse import unquote_plus
from pymongo.client_options import _parse_ssl_options
from pymongo.common import (
from pymongo.errors import ConfigurationError, InvalidURI
from pymongo.srv_resolver import _HAVE_DNSPYTHON, _SrvResolver
from pymongo.typings import _Address
def _normalize_options(options: _CaseInsensitiveDictionary) -> _CaseInsensitiveDictionary:
    """Normalizes option names in the options dictionary by converting them to
    their internally-used names.

    :Parameters:
        - `options`: Instance of _CaseInsensitiveDictionary containing
          MongoDB URI options.
    """
    tlsinsecure = options.get('tlsinsecure')
    if tlsinsecure is not None:
        for opt in _IMPLICIT_TLSINSECURE_OPTS:
            options[opt] = tlsinsecure
    for optname in list(options):
        intname = INTERNAL_URI_OPTION_NAME_MAP.get(optname, None)
        if intname is not None:
            options[intname] = options.pop(optname)
    return options