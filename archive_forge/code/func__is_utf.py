import os
import re
import sys
from functools import partial, partialmethod, wraps
from inspect import signature
from unicodedata import east_asian_width
from warnings import warn
from weakref import proxy
def _is_utf(encoding):
    try:
        u'█▉'.encode(encoding)
    except UnicodeEncodeError:
        return False
    except Exception:
        try:
            return encoding.lower().startswith('utf-') or 'U8' == encoding
        except Exception:
            return False
    else:
        return True