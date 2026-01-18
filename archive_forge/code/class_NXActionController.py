import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class NXActionController(NXAction):
    """
        Send packet in message action

        This action sends the packet to the OpenFlow controller as
        a packet in message.

        And equivalent to the followings action of ovs-ofctl command.

        ..
          controller(key=value...)
        ..

        +----------------------------------------------+
        | **controller(**\\ *key*\\=\\ *value*\\...\\ **)** |
        +----------------------------------------------+

        ================ ======================================================
        Attribute        Description
        ================ ======================================================
        max_len          Max length to send to controller
        controller_id    Controller ID to send packet-in
        reason           Reason for sending the message
        ================ ======================================================

        Example::

            actions += [
                parser.NXActionController(max_len=1024,
                                          controller_id=1,
                                          reason=ofproto.OFPR_INVALID_TTL)]
        """
    _subtype = nicira_ext.NXAST_CONTROLLER
    _fmt_str = '!HHBx'

    def __init__(self, max_len, controller_id, reason, type_=None, len_=None, experimenter=None, subtype=None):
        super(NXActionController, self).__init__()
        self.max_len = max_len
        self.controller_id = controller_id
        self.reason = reason

    @classmethod
    def parser(cls, buf):
        max_len, controller_id, reason = struct.unpack_from(cls._fmt_str, buf)
        return cls(max_len, controller_id, reason)

    def serialize_body(self):
        data = bytearray()
        msg_pack_into(self._fmt_str, data, 0, self.max_len, self.controller_id, self.reason)
        return data