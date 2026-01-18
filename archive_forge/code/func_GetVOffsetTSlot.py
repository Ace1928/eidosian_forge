from . import encode
from . import number_types as N
def GetVOffsetTSlot(self, slot, d):
    """
        GetVOffsetTSlot retrieves the VOffsetT that the given vtable location
        points to. If the vtable value is zero, the default value `d`
        will be returned.
        """
    N.enforce_number(slot, N.VOffsetTFlags)
    N.enforce_number(d, N.VOffsetTFlags)
    off = self.Offset(slot)
    if off == 0:
        return d
    return off