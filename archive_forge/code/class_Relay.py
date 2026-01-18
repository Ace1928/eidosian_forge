import struct
import dns.exception
import dns.immutable
import dns.rdtypes.util
class Relay(dns.rdtypes.util.Gateway):
    name = 'AMTRELAY relay'

    @property
    def relay(self):
        return self.gateway