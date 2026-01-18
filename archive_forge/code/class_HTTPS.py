import dns.immutable
import dns.rdtypes.svcbbase
@dns.immutable.immutable
class HTTPS(dns.rdtypes.svcbbase.SVCBBase):
    """HTTPS record"""