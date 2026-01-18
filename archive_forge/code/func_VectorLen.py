from . import encode
from . import number_types as N
def VectorLen(self, off):
    """VectorLen retrieves the length of the vector whose offset is stored
           at "off" in this object."""
    N.enforce_number(off, N.UOffsetTFlags)
    off += self.Pos
    off += encode.Get(N.UOffsetTFlags.packer_type, self.Bytes, off)
    ret = encode.Get(N.UOffsetTFlags.packer_type, self.Bytes, off)
    return ret