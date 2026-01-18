import dns.enum
class NSEC3Hash(dns.enum.IntEnum):
    """NSEC3 hash algorithm"""
    SHA1 = 1

    @classmethod
    def _maximum(cls):
        return 255