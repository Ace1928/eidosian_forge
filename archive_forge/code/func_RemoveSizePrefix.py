from . import encode
from . import number_types
from . import packer
def RemoveSizePrefix(buf, offset):
    """
	Create a slice of a size-prefixed buffer that has
	its position advanced just past the size prefix.
	"""
    return (buf, offset + number_types.Int32Flags.bytewidth)