import abc
import struct
from . import packet_base
from . import icmpv6
from . import tcp
from . import udp
from . import sctp
from . import gre
from . import in_proto as inet
from os_ken.lib import addrconv
from os_ken.lib import stringify
@ipv6.register_header_type(inet.IPPROTO_ROUTING)
class routing(header):
    """An IPv6 Routing Header decoder class.
    This class has only the parser method.

    IPv6 Routing Header types.

    http://www.iana.org/assignments/ipv6-parameters/ipv6-parameters.xhtml

    +-----------+----------------------------------+-------------------+
    | Value     | Description                      | Reference         |
    +===========+==================================+===================+
    | 0         | Source Route (DEPRECATED)        | [[IPV6]][RFC5095] |
    +-----------+----------------------------------+-------------------+
    | 1         | Nimrod (DEPRECATED 2009-05-06)   |                   |
    +-----------+----------------------------------+-------------------+
    | 2         | Type 2 Routing Header            | [RFC6275]         |
    +-----------+----------------------------------+-------------------+
    | 3         | RPL Source Route Header          | [RFC6554]         |
    +-----------+----------------------------------+-------------------+
    | 4 - 252   | Unassigned                       |                   |
    +-----------+----------------------------------+-------------------+
    | 253       | RFC3692-style Experiment 1 [2]   | [RFC4727]         |
    +-----------+----------------------------------+-------------------+
    | 254       | RFC3692-style Experiment 2 [2]   | [RFC4727]         |
    +-----------+----------------------------------+-------------------+
    | 255       | Reserved                         |                   |
    +-----------+----------------------------------+-------------------+
    """
    TYPE = inet.IPPROTO_ROUTING
    _OFFSET_LEN = struct.calcsize('!2B')
    ROUTING_TYPE_2 = 2
    ROUTING_TYPE_3 = 3

    @classmethod
    def parser(cls, buf):
        type_, = struct.unpack_from('!B', buf, cls._OFFSET_LEN)
        switch = {cls.ROUTING_TYPE_2: None, cls.ROUTING_TYPE_3: routing_type3}
        cls_ = switch.get(type_)
        if cls_:
            return cls_.parser(buf)
        else:
            return None