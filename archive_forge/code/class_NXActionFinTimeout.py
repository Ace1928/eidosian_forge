import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class NXActionFinTimeout(NXAction):
    """
        Change TCP timeout action

        This action changes the idle timeout or hard timeout or
        both, of this OpenFlow rule when the rule matches a TCP
        packet with the FIN or RST flag.

        And equivalent to the followings action of ovs-ofctl command.

        ..
          fin_timeout(argument[,argument]...)
        ..

        +---------------------------------------------------------+
        | **fin_timeout(**\\ *argument*\\[,\\ *argument*\\]...\\ **)** |
        +---------------------------------------------------------+

        ================ ======================================================
        Attribute        Description
        ================ ======================================================
        fin_idle_timeout Causes the flow to expire after the given number
                         of seconds of inactivity
        fin_idle_timeout Causes the flow to expire after the given number
                         of second, regardless of activity
        ================ ======================================================

        Example::

            match = parser.OFPMatch(ip_proto=6, eth_type=0x0800)
            actions += [parser.NXActionFinTimeout(fin_idle_timeout=30,
                                                  fin_hard_timeout=60)]
        """
    _subtype = nicira_ext.NXAST_FIN_TIMEOUT
    _fmt_str = '!HH2x'

    def __init__(self, fin_idle_timeout, fin_hard_timeout, type_=None, len_=None, experimenter=None, subtype=None):
        super(NXActionFinTimeout, self).__init__()
        self.fin_idle_timeout = fin_idle_timeout
        self.fin_hard_timeout = fin_hard_timeout

    @classmethod
    def parser(cls, buf):
        fin_idle_timeout, fin_hard_timeout = struct.unpack_from(cls._fmt_str, buf, 0)
        return cls(fin_idle_timeout, fin_hard_timeout)

    def serialize_body(self):
        data = bytearray()
        msg_pack_into(self._fmt_str, data, 0, self.fin_idle_timeout, self.fin_hard_timeout)
        return data