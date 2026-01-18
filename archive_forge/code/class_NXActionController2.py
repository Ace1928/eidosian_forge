import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class NXActionController2(NXAction):
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
        userdata         Additional data to the controller in the packet-in
                         message
        pause            Flag to pause pipeline to resume later
        ================ ======================================================

        Example::

            actions += [
                parser.NXActionController(max_len=1024,
                                          controller_id=1,
                                          reason=ofproto.OFPR_INVALID_TTL,
                                          userdata=[0xa,0xb,0xc],
                                          pause=True)]
        """
    _subtype = nicira_ext.NXAST_CONTROLLER2
    _fmt_str = '!6x'
    _PACK_STR = '!HH'

    def __init__(self, type_=None, len_=None, vendor=None, subtype=None, **kwargs):
        super(NXActionController2, self).__init__()
        for arg in kwargs:
            if arg in NXActionController2Prop._NAMES:
                setattr(self, arg, kwargs[arg])

    @classmethod
    def parser(cls, buf):
        cls_data = {}
        offset = 6
        buf_len = len(buf)
        while buf_len > offset:
            type_, length = struct.unpack_from(cls._PACK_STR, buf, offset)
            offset += 4
            try:
                subcls = NXActionController2Prop._TYPES[type_]
            except KeyError:
                subcls = NXActionController2PropUnknown
            data, size = subcls.parser_prop(buf[offset:], length - 4)
            offset += size
            cls_data[subcls._arg_name] = data
        return cls(**cls_data)

    def serialize_body(self):
        body = bytearray()
        msg_pack_into(self._fmt_str, body, 0)
        prop_list = []
        for arg in self.__dict__:
            if arg in NXActionController2Prop._NAMES:
                prop_list.append((NXActionController2Prop._NAMES[arg], self.__dict__[arg]))
        prop_list.sort(key=lambda x: x[0].type)
        for subcls, value in prop_list:
            body += subcls.serialize_prop(value)
        return body