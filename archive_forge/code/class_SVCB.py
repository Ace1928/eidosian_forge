import dns.immutable
import dns.rdtypes.svcbbase
@dns.immutable.immutable
class SVCB(dns.rdtypes.svcbbase.SVCBBase):
    """SVCB record"""