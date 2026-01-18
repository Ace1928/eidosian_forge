from __future__ import unicode_literals
from past.builtins import basestring
from .dag import KwargReprNode
from ._utils import escape_chars, get_hash_int
from builtins import object
import os
def _is_of_types(obj, types):
    valid = False
    for stream_type in types:
        if isinstance(obj, stream_type):
            valid = True
            break
    return valid