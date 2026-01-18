import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class atomic(Stub):
    """Namespace for atomic operations
    """
    _description_ = '<atomic>'

    class add(Stub):
        """add(ary, idx, val)

        Perform atomic ``ary[idx] += val``. Supported on int32, float32, and
        float64 operands only.

        Returns the old value at the index location as if it is loaded
        atomically.
        """

    class sub(Stub):
        """sub(ary, idx, val)

        Perform atomic ``ary[idx] -= val``. Supported on int32, float32, and
        float64 operands only.

        Returns the old value at the index location as if it is loaded
        atomically.
        """

    class and_(Stub):
        """and_(ary, idx, val)

        Perform atomic ``ary[idx] &= val``. Supported on int32, int64, uint32
        and uint64 operands only.

        Returns the old value at the index location as if it is loaded
        atomically.
        """

    class or_(Stub):
        """or_(ary, idx, val)

        Perform atomic ``ary[idx] |= val``. Supported on int32, int64, uint32
        and uint64 operands only.

        Returns the old value at the index location as if it is loaded
        atomically.
        """

    class xor(Stub):
        """xor(ary, idx, val)

        Perform atomic ``ary[idx] ^= val``. Supported on int32, int64, uint32
        and uint64 operands only.

        Returns the old value at the index location as if it is loaded
        atomically.
        """

    class inc(Stub):
        """inc(ary, idx, val)

        Perform atomic ``ary[idx] += 1`` up to val, then reset to 0. Supported
        on uint32, and uint64 operands only.

        Returns the old value at the index location as if it is loaded
        atomically.
        """

    class dec(Stub):
        """dec(ary, idx, val)

        Performs::

           ary[idx] = (value if (array[idx] == 0) or
                       (array[idx] > value) else array[idx] - 1)

        Supported on uint32, and uint64 operands only.

        Returns the old value at the index location as if it is loaded
        atomically.
        """

    class exch(Stub):
        """exch(ary, idx, val)

        Perform atomic ``ary[idx] = val``. Supported on int32, int64, uint32 and
        uint64 operands only.

        Returns the old value at the index location as if it is loaded
        atomically.
        """

    class max(Stub):
        """max(ary, idx, val)

        Perform atomic ``ary[idx] = max(ary[idx], val)``.

        Supported on int32, int64, uint32, uint64, float32, float64 operands
        only.

        Returns the old value at the index location as if it is loaded
        atomically.
        """

    class min(Stub):
        """min(ary, idx, val)

        Perform atomic ``ary[idx] = min(ary[idx], val)``.

        Supported on int32, int64, uint32, uint64, float32, float64 operands
        only.

        Returns the old value at the index location as if it is loaded
        atomically.
        """

    class nanmax(Stub):
        """nanmax(ary, idx, val)

        Perform atomic ``ary[idx] = max(ary[idx], val)``.

        NOTE: NaN is treated as a missing value such that:
        nanmax(NaN, n) == n, nanmax(n, NaN) == n

        Supported on int32, int64, uint32, uint64, float32, float64 operands
        only.

        Returns the old value at the index location as if it is loaded
        atomically.
        """

    class nanmin(Stub):
        """nanmin(ary, idx, val)

        Perform atomic ``ary[idx] = min(ary[idx], val)``.

        NOTE: NaN is treated as a missing value, such that:
        nanmin(NaN, n) == n, nanmin(n, NaN) == n

        Supported on int32, int64, uint32, uint64, float32, float64 operands
        only.

        Returns the old value at the index location as if it is loaded
        atomically.
        """

    class compare_and_swap(Stub):
        """compare_and_swap(ary, old, val)

        Conditionally assign ``val`` to the first element of an 1D array ``ary``
        if the current value matches ``old``.

        Supported on int32, int64, uint32, uint64 operands only.

        Returns the old value as if it is loaded atomically.
        """

    class cas(Stub):
        """cas(ary, idx, old, val)

        Conditionally assign ``val`` to the element ``idx`` of an array
        ``ary`` if the current value of ``ary[idx]`` matches ``old``.

        Supported on int32, int64, uint32, uint64 operands only.

        Returns the old value as if it is loaded atomically.
        """