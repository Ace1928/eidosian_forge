from __future__ import absolute_import, division, print_function
from base64 import (
from binascii import b2a_base64, a2b_base64, Error as _BinAsciiError
import logging
from passlib import exc
from passlib.utils.compat import (
from passlib.utils.decor import memoized_property
def __make_padset(self, bits):
    """helper to generate set of valid last chars & bytes"""
    pset = set((c for i, c in enumerate(self.bytemap) if not i & bits))
    pset.update((c for i, c in enumerate(self.charmap) if not i & bits))
    return frozenset(pset)