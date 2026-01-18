from . import encode
from . import number_types as N
def Indirect(self, off):
    """Indirect retrieves the relative offset stored at `offset`."""
    N.enforce_number(off, N.UOffsetTFlags)
    return off + encode.Get(N.UOffsetTFlags.packer_type, self.Bytes, off)