import struct
import dns.edns
import dns.exception
import dns.immutable
import dns.rdata
@dns.immutable.immutable
class OPT(dns.rdata.Rdata):
    """OPT record"""
    __slots__ = ['options']

    def __init__(self, rdclass, rdtype, options):
        """Initialize an OPT rdata.

        *rdclass*, an ``int`` is the rdataclass of the Rdata,
        which is also the payload size.

        *rdtype*, an ``int`` is the rdatatype of the Rdata.

        *options*, a tuple of ``bytes``
        """
        super().__init__(rdclass, rdtype)

        def as_option(option):
            if not isinstance(option, dns.edns.Option):
                raise ValueError('option is not a dns.edns.option')
            return option
        self.options = self._as_tuple(options, as_option)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        for opt in self.options:
            owire = opt.to_wire()
            file.write(struct.pack('!HH', opt.otype, len(owire)))
            file.write(owire)

    def to_text(self, origin=None, relativize=True, **kw):
        return ' '.join((opt.to_text() for opt in self.options))

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        options = []
        while parser.remaining() > 0:
            otype, olen = parser.get_struct('!HH')
            with parser.restrict_to(olen):
                opt = dns.edns.option_from_wire_parser(otype, parser)
            options.append(opt)
        return cls(rdclass, rdtype, options)

    @property
    def payload(self):
        """payload size"""
        return self.rdclass