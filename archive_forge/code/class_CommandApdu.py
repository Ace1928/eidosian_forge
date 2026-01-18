import struct
from pyu2f import errors
class CommandApdu(object):
    """Represents a Command APDU.

  Represents a Command APDU sent to the security key.  Encoding
  is specified in FIDO U2F standards.
  """
    cla = None
    ins = None
    p1 = None
    p2 = None
    data = None

    def __init__(self, cla, ins, p1, p2, data=None):
        self.cla = cla
        self.ins = ins
        self.p1 = p1
        self.p2 = p2
        if data and len(data) > 65535:
            raise errors.InvalidCommandError()
        if data:
            self.data = data

    def ToByteArray(self):
        """Serialize the command.

    Encodes the command as per the U2F specs, using the standard
    ISO 7816-4 extended encoding.  All Commands expect data, so
    Le is always present.

    Returns:
      Python bytearray of the encoded command.
    """
        lc = self.InternalEncodeLc()
        out = bytearray(4)
        out[0] = self.cla
        out[1] = self.ins
        out[2] = self.p1
        out[3] = self.p2
        if self.data:
            out.extend(lc)
            out.extend(self.data)
            out.extend([0, 0])
        else:
            out.extend([0, 0, 0])
        return out

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

    def InternalEncodeLc(self):
        dl = 0
        if self.data:
            dl = len(self.data)
        fourbyte = struct.pack('>I', dl)
        return bytearray(fourbyte[1:])