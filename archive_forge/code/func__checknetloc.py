from collections import namedtuple
import functools
import re
import sys
import types
import warnings
import ipaddress
def _checknetloc(netloc):
    if not netloc or netloc.isascii():
        return
    import unicodedata
    n = netloc.replace('@', '')
    n = n.replace(':', '')
    n = n.replace('#', '')
    n = n.replace('?', '')
    netloc2 = unicodedata.normalize('NFKC', n)
    if n == netloc2:
        return
    for c in '/?#@:':
        if c in netloc2:
            raise ValueError("netloc '" + netloc + "' contains invalid " + 'characters under NFKC normalization')