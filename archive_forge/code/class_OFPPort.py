import struct
import base64
from os_ken.lib import addrconv
from os_ken.lib import mac
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.lib.packet import packet
from os_ken import exception
from os_ken import utils
from os_ken.ofproto.ofproto_parser import StringifyMixin, MsgBase
from os_ken.ofproto import ether
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import nx_actions
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_common
from os_ken.ofproto import ofproto_v1_3 as ofproto
import logging
class OFPPort(ofproto_parser.namedtuple('OFPPort', ('port_no', 'hw_addr', 'name', 'config', 'state', 'curr', 'advertised', 'supported', 'peer', 'curr_speed', 'max_speed'))):
    """
    Description of a port

    ========== =========================================================
    Attribute  Description
    ========== =========================================================
    port_no    Port number and it uniquely identifies a port within
               a switch.
    hw_addr    MAC address for the port.
    name       Null-terminated string containing a human-readable name
               for the interface.
    config     Bitmap of port configuration flags.

               | OFPPC_PORT_DOWN
               | OFPPC_NO_RECV
               | OFPPC_NO_FWD
               | OFPPC_NO_PACKET_IN
    state      Bitmap of port state flags.

               | OFPPS_LINK_DOWN
               | OFPPS_BLOCKED
               | OFPPS_LIVE
    curr       Current features.
    advertised Features being advertised by the port.
    supported  Features supported by the port.
    peer       Features advertised by peer.
    curr_speed Current port bitrate in kbps.
    max_speed  Max port bitrate in kbps.
    ========== =========================================================
    """
    _TYPE = {'ascii': ['hw_addr'], 'utf-8': ['name']}

    @classmethod
    def parser(cls, buf, offset):
        port = struct.unpack_from(ofproto.OFP_PORT_PACK_STR, buf, offset)
        port = list(port)
        i = cls._fields.index('hw_addr')
        port[i] = addrconv.mac.bin_to_text(port[i])
        i = cls._fields.index('name')
        port[i] = port[i].rstrip(b'\x00')
        ofpport = cls(*port)
        ofpport.length = ofproto.OFP_PORT_SIZE
        return ofpport