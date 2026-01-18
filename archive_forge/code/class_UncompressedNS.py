from io import BytesIO
import dns.exception
import dns.rdata
import dns.name
class UncompressedNS(NSBase):
    """Base class for rdata that is like an NS record, but whose name
    is not compressed when convert to DNS wire format, and whose
    digestable form is not downcased."""

    def to_wire(self, file, compress=None, origin=None):
        super(UncompressedNS, self).to_wire(file, None, origin)

    def to_digestable(self, origin=None):
        f = BytesIO()
        self.to_wire(f, None, origin)
        return f.getvalue()