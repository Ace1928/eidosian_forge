from ..construct import (
def SLEB128(name):
    """ A construct creator for SLEB128 encoding.
    """
    return Rename(name, _SLEB128Adapter(_LEB128_reader()))