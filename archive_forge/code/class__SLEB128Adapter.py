from ..construct import (
class _SLEB128Adapter(Adapter):
    """ An adapter for SLEB128, given a sequence of bytes in a sub-construct.
    """

    def _decode(self, obj, context):
        value = 0
        for b in reversed(obj):
            value = (value << 7) + (ord(b) & 127)
        if ord(obj[-1]) & 64:
            value |= -(1 << 7 * len(obj))
        return value