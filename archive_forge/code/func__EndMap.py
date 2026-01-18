import array
import contextlib
import enum
import struct
def _EndMap(self, start):
    """Finishes map construction by encodung its elements."""
    stack = self._stack[start:]
    if len(stack) % 2 != 0:
        raise RuntimeError('must be even number of keys and values')
    for key in stack[::2]:
        if key.Type is not Type.KEY:
            raise RuntimeError('all map keys must be of %s type' % Type.KEY)
    pairs = zip(stack[::2], stack[1::2])
    pairs = sorted(pairs, key=lambda pair: self._ReadKey(pair[0].Value))
    del self._stack[start:]
    for pair in pairs:
        self._stack.extend(pair)
    keys = self._CreateVector(self._stack[start::2], typed=True, fixed=False)
    values = self._CreateVector(self._stack[start + 1::2], typed=False, fixed=False, keys=keys)
    del self._stack[start:]
    self._stack.append(values)
    return values.Value