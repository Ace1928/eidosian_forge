import struct
import base64
from os_ken.lib import addrconv
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.lib.packet import packet
from os_ken import exception
from os_ken import utils
from os_ken.ofproto.ofproto_parser import StringifyMixin, MsgBase, MsgInMsgBase
from os_ken.ofproto import ether
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import nx_actions
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_common
from os_ken.ofproto import ofproto_v1_5 as ofproto
class OFPStats(StringifyMixin):
    """
    Flow Stats Structure

    This class is implementation of the flow stats structure having
    compose/query API.

    You can define the flow stats by the keyword arguments.
    The following arguments are available.

    ============= ================ ============================================
    Argument      Value            Description
    ============= ================ ============================================
    duration      Integer 32bit*2  Time flow entry has been alive. This field
                                   is a tuple of two Integer 32bit. The first
                                   value is duration_sec and the second is
                                   duration_nsec.
    idle_time     Integer 32bit*2  Time flow entry has been idle.
    flow_count    Integer 32bit    Number of aggregated flow entries.
    packet_count  Integer 64bit    Number of packets matched by a flow entry.
    byte_count    Integer 64bit    Number of bytes matched by a flow entry.
    ============= ================ ============================================

    Example::

        >>> # compose
        >>> stats = parser.OFPStats(
        ...     packet_count=100,
        ...     duration=(100, 200)
        >>> # query
        >>> if 'duration' in stats:
        ...     print stats['duration']
        ...
        (100, 200)
    """

    def __init__(self, length=None, _ordered_fields=None, **kwargs):
        super(OFPStats, self).__init__()
        self.length = length
        if _ordered_fields is not None:
            assert not kwargs
            self.fields = _ordered_fields
        else:
            fields = [ofproto.oxs_from_user(k, v) for k, v in kwargs.items()]
            fields.sort(key=lambda x: x[0][0] if isinstance(x[0], tuple) else x[0])
            self.fields = [ofproto.oxs_to_user(n, v, None) for n, v, _ in fields]

    @classmethod
    def parser(cls, buf, offset):
        """
        Returns an object which is generated from a buffer including the
        expression of the wire protocol of the flow stats.
        """
        stats = OFPStats()
        reserved, length = struct.unpack_from('!HH', buf, offset)
        stats.length = length
        offset += 4
        length -= 4
        fields = []
        while length > 0:
            n, value, _, field_len = ofproto.oxs_parse(buf, offset)
            k, uv = ofproto.oxs_to_user(n, value, None)
            fields.append((k, uv))
            offset += field_len
            length -= field_len
        stats.fields = fields
        return stats

    def serialize(self, buf, offset):
        """
        Outputs the expression of the wire protocol of the flow stats into
        the buf.
        Returns the output length.
        """
        fields = [ofproto.oxs_from_user(k, uv) for k, uv in self.fields]
        hdr_pack_str = '!HH'
        field_offset = offset + struct.calcsize(hdr_pack_str)
        for n, value, _ in fields:
            field_offset += ofproto.oxs_serialize(n, value, None, buf, field_offset)
        reserved = 0
        length = field_offset - offset
        msg_pack_into(hdr_pack_str, buf, offset, reserved, length)
        self.length = length
        pad_len = utils.round_up(length, 8) - length
        msg_pack_into('%dx' % pad_len, buf, field_offset)
        return length + pad_len

    def __getitem__(self, key):
        return dict(self.fields)[key]

    def __contains__(self, key):
        return key in dict(self.fields)

    def iteritems(self):
        return dict(self.fields).items()

    def items(self):
        return self.fields

    def get(self, key, default=None):
        return dict(self.fields).get(key, default)

    def stringify_attrs(self):
        yield ('oxs_fields', dict(self.fields))

    def to_jsondict(self):
        """
        Returns a dict expressing the flow stats.
        """
        body = {'oxs_fields': [ofproto.oxs_to_jsondict(k, uv) for k, uv in self.fields], 'length': self.length}
        return {self.__class__.__name__: body}

    @classmethod
    def from_jsondict(cls, dict_):
        """
        Returns an object which is generated from a dict.

        Exception raises:
        KeyError -- Unknown stats field is defined in dict
        """
        fields = [ofproto.oxs_from_jsondict(f) for f in dict_['oxs_fields']]
        return OFPStats(_ordered_fields=fields)