import struct
from pyu2f import errors
def ToLegacyU2FByteArray(self):
    """Serialize the command in the legacy format.

    Encodes the command as per the U2F specs, using the legacy
    encoding in which LC is always present.

    Returns:
      Python bytearray of the encoded command.
    """
    lc = self.InternalEncodeLc()
    out = bytearray(4)
    out[0] = self.cla
    out[1] = self.ins
    out[2] = self.p1
    out[3] = self.p2
    out.extend(lc)
    if self.data:
        out.extend(self.data)
    out.extend([0, 0])
    return out