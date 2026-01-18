import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class NXActionRegLoad2(NXAction):
    """
        Load literal value action

        This action loads a literal value into a field or part of a field.

        And equivalent to the followings action of ovs-ofctl command.

        ..
          set_field:value[/mask]->dst
        ..

        +------------------------------------------------------------+
        | **set_field**\\:\\ *value*\\ **[**\\/\\ *mask*\\ **]**\\->\\ *dst* |
        +------------------------------------------------------------+

        ================ ======================================================
        Attribute        Description
        ================ ======================================================
        value            OXM/NXM value to be loaded
        mask             Mask for destination field
        dst              OXM/NXM header for destination field
        ================ ======================================================

        Example::

            actions += [parser.NXActionRegLoad2(dst="tun_ipv4_src",
                                                value="192.168.10.0",
                                                mask="255.255.255.0")]
        """
    _subtype = nicira_ext.NXAST_REG_LOAD2
    _TYPE = {'ascii': ['dst', 'value']}

    def __init__(self, dst, value, mask=None, type_=None, len_=None, experimenter=None, subtype=None):
        super(NXActionRegLoad2, self).__init__()
        self.dst = dst
        self.value = value
        self.mask = mask

    @classmethod
    def parser(cls, buf):
        n, uv, mask, _len = ofp.oxm_parse(buf, 0)
        dst, value = ofp.oxm_to_user(n, uv, mask)
        if isinstance(value, (tuple, list)):
            return cls(dst, value[0], value[1])
        else:
            return cls(dst, value, None)

    def serialize_body(self):
        data = bytearray()
        if self.mask is None:
            value = self.value
        else:
            value = (self.value, self.mask)
            self._TYPE['ascii'].append('mask')
        n, value, mask = ofp.oxm_from_user(self.dst, value)
        len_ = ofp.oxm_serialize(n, value, mask, data, 0)
        msg_pack_into('!%dx' % (14 - len_), data, len_)
        return data