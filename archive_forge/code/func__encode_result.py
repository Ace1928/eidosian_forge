from collections import namedtuple
import functools
import re
import sys
import types
import warnings
import ipaddress
def _encode_result(obj, encoding=_implicit_encoding, errors=_implicit_errors):
    return obj.encode(encoding, errors)