from __future__ import unicode_literals
from past.builtins import basestring
from .dag import KwargReprNode
from ._utils import escape_chars, get_hash_int
from builtins import object
import os
def _get_types_str(types):
    return ', '.join(['{}.{}'.format(x.__module__, x.__name__) for x in types])