import logging
import logging.handlers
import sys
from tornado.escape import _unicode
from tornado.util import unicode_type, basestring_type
from typing import Dict, Any, cast, Optional
def _safe_unicode(s: Any) -> str:
    try:
        return _unicode(s)
    except UnicodeDecodeError:
        return repr(s)