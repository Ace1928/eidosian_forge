from __future__ import with_statement, division
from functools import partial
from passlib.utils import getrandbytes
from passlib.tests.utils import TestCase
def _random_parity(self, key):
    """randomize parity bits"""
    from passlib.crypto.des import _KDATA_MASK, _KPARITY_MASK, INT_64_MASK
    rng = self.getRandom()
    return key & _KDATA_MASK | rng.randint(0, INT_64_MASK) & _KPARITY_MASK