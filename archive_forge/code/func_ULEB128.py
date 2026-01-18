from ..construct import (
def ULEB128(name):
    """ A construct creator for ULEB128 encoding.
    """
    return Rename(name, _ULEB128Adapter(_LEB128_reader()))