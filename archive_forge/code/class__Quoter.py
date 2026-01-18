from collections import namedtuple
import functools
import re
import sys
import types
import warnings
import ipaddress
class _Quoter(dict):
    """A mapping from bytes numbers (in range(0,256)) to strings.

    String values are percent-encoded byte values, unless the key < 128, and
    in either of the specified safe set, or the always safe set.
    """

    def __init__(self, safe):
        """safe: bytes object."""
        self.safe = _ALWAYS_SAFE.union(safe)

    def __repr__(self):
        return f'<Quoter {dict(self)!r}>'

    def __missing__(self, b):
        res = chr(b) if b in self.safe else '%{:02X}'.format(b)
        self[b] = res
        return res