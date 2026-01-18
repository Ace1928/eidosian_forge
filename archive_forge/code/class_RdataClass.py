import dns.enum
import dns.exception
class RdataClass(dns.enum.IntEnum):
    """DNS Rdata Class"""
    RESERVED0 = 0
    IN = 1
    INTERNET = IN
    CH = 3
    CHAOS = CH
    HS = 4
    HESIOD = HS
    NONE = 254
    ANY = 255

    @classmethod
    def _maximum(cls):
        return 65535

    @classmethod
    def _short_name(cls):
        return 'class'

    @classmethod
    def _prefix(cls):
        return 'CLASS'

    @classmethod
    def _unknown_exception_class(cls):
        return UnknownRdataclass