from __future__ import with_statement, absolute_import
import logging
import re
import types
from warnings import warn
from passlib import exc
from passlib.crypto.digest import MAX_UINT32
from passlib.utils import classproperty, to_bytes, render_bytes
from passlib.utils.binary import b64s_encode, b64s_decode
from passlib.utils.compat import u, unicode, bascii_to_str, uascii_to_str, PY2
import passlib.utils.handlers as uh
@classmethod
def _norm_memory_cost(cls, memory_cost, relaxed=False):
    return uh.norm_integer(cls, memory_cost, min=cls.min_memory_cost, param='memory_cost', relaxed=relaxed)