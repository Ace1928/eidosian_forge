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
class SystemRandom(Random):
    """Alternate random number generator using sources provided
    by the operating system (such as /dev/urandom on Unix or
    CryptGenRandom on Windows).

     Not available on all systems (see os.urandom() for details).

    """

    def random(self):
        """Get the next random number in the range 0.0 <= X < 1.0."""
        return (int.from_bytes(_urandom(7)) >> 3) * RECIP_BPF

    def getrandbits(self, k):
        """getrandbits(k) -> x.  Generates an int with k random bits."""
        if k < 0:
            raise ValueError('number of bits must be non-negative')
        numbytes = (k + 7) // 8
        x = int.from_bytes(_urandom(numbytes))
        return x >> numbytes * 8 - k

    def randbytes(self, n):
        """Generate n random bytes."""
        return _urandom(n)

    def seed(self, *args, **kwds):
        """Stub method.  Not used for a system random number generator."""
        return None

    def _notimplemented(self, *args, **kwds):
        """Method should not be called for a system random number generator."""
        raise NotImplementedError('System entropy source does not have state.')
    getstate = setstate = _notimplemented