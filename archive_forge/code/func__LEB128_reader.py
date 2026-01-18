from ..construct import (
def _LEB128_reader():
    """ Read LEB128 variable-length data from the stream. The data is terminated
        by a byte with 0 in its highest bit.
    """
    return RepeatUntil(lambda obj, ctx: ord(obj) < 128, Field(None, 1))