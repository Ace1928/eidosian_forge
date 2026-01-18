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
def _unquoted_percent(s: str) -> bool:
    """Check for unescaped percent signs.

    :Parameters:
        - `s`: A string. `s` can have things like '%25', '%2525',
           and '%E2%85%A8' but cannot have unquoted percent like '%foo'.
    """
    for i in range(len(s)):
        if s[i] == '%':
            sub = s[i:i + 3]
            if unquote_plus(sub) == sub:
                return True
    return False