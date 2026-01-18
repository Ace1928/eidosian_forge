import struct
from . import packet_base
from os_ken.lib import addrconv
class slow(packet_base.PacketBase):
    """Slow Protocol header decoder class.
    This class has only the parser method.

    http://standards.ieee.org/getieee802/download/802.3-2012_section5.pdf

    Slow Protocols Subtypes

    +---------------+--------------------------------------------------+
    | Subtype Value | Protocol Name                                    |
    +===============+==================================================+
    | 0             | Unused - Illegal Value                           |
    +---------------+--------------------------------------------------+
    | 1             | Link Aggregation Control Protocol(LACP)          |
    +---------------+--------------------------------------------------+
    | 2             | Link Aggregation - Marker Protocol               |
    +---------------+--------------------------------------------------+
    | 3             | Operations, Administration, and Maintenance(OAM) |
    +---------------+--------------------------------------------------+
    | 4 - 9         | Reserved for future use                          |
    +---------------+--------------------------------------------------+
    | 10            | Organization Specific Slow Protocol(OSSP)        |
    +---------------+--------------------------------------------------+
    | 11 - 255      | Unused - Illegal values                          |
    +---------------+--------------------------------------------------+
    """
    _PACK_STR = '!B'

    @classmethod
    def parser(cls, buf):
        subtype, = struct.unpack_from(cls._PACK_STR, buf)
        switch = {SLOW_SUBTYPE_LACP: lacp, SLOW_SUBTYPE_MARKER: None, SLOW_SUBTYPE_OAM: None, SLOW_SUBTYPE_OSSP: None}
        cls_ = switch.get(subtype)
        if cls_:
            return cls_.parser(buf)
        else:
            return (None, None, buf)