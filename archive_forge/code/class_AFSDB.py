import dns.rdtypes.mxbase
class AFSDB(dns.rdtypes.mxbase.UncompressedDowncasingMX):
    """AFSDB record

    @ivar subtype: the subtype value
    @type subtype: int
    @ivar hostname: the hostname name
    @type hostname: dns.name.Name object"""

    def get_subtype(self):
        return self.preference

    def set_subtype(self, subtype):
        self.preference = subtype
    subtype = property(get_subtype, set_subtype)

    def get_hostname(self):
        return self.exchange

    def set_hostname(self, hostname):
        self.exchange = hostname
    hostname = property(get_hostname, set_hostname)