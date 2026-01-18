from io import BytesIO
import struct
import dns.exception
import dns.rdata
import dns.name
class UncompressedDowncasingMX(MXBase):
    """Base class for rdata that is like an MX record, but whose name
    is not compressed when convert to DNS wire format."""

    def to_wire(self, file, compress=None, origin=None):
        super(UncompressedDowncasingMX, self).to_wire(file, None, origin)