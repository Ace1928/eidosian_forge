from . import encode
from . import number_types as N
def Vector(self, off):
    """Vector retrieves the start of data of the vector whose offset is
           stored at "off" in this object."""
    N.enforce_number(off, N.UOffsetTFlags)
    off += self.Pos
    x = off + self.Get(N.UOffsetTFlags, off)
    x += N.UOffsetTFlags.bytewidth
    return x