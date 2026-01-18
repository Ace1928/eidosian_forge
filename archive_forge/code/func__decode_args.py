from collections import namedtuple
import functools
import re
import sys
import types
import warnings
import ipaddress
def _decode_args(args, encoding=_implicit_encoding, errors=_implicit_errors):
    return tuple((x.decode(encoding, errors) if x else '' for x in args))