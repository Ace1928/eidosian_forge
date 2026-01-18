from collections import namedtuple
import functools
import re
import sys
import types
import warnings
import ipaddress
@functools.lru_cache
def _byte_quoter_factory(safe):
    return _Quoter(safe).__getitem__