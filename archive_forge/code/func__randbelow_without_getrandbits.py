from warnings import warn as _warn
from math import log as _log, exp as _exp, pi as _pi, e as _e, ceil as _ceil
from math import sqrt as _sqrt, acos as _acos, cos as _cos, sin as _sin
from math import tau as TWOPI, floor as _floor, isfinite as _isfinite
from os import urandom as _urandom
from _collections_abc import Set as _Set, Sequence as _Sequence
from operator import index as _index
from itertools import accumulate as _accumulate, repeat as _repeat
from bisect import bisect as _bisect
import os as _os
import _random
def _randbelow_without_getrandbits(self, n, maxsize=1 << BPF):
    """Return a random int in the range [0,n).  Defined for n > 0.

        The implementation does not use getrandbits, but only random.
        """
    random = self.random
    if n >= maxsize:
        _warn('Underlying random() generator does not supply \nenough bits to choose from a population range this large.\nTo remove the range limitation, add a getrandbits() method.')
        return _floor(random() * n)
    rem = maxsize % n
    limit = (maxsize - rem) / maxsize
    r = random()
    while r >= limit:
        r = random()
    return _floor(r * maxsize) % n