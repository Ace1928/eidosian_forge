import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import stream_parser
from os_ken.lib.packet.bgp import BGPMessage
from os_ken.lib.type_desc import TypeDisp
@BMPMessage.register_type(BMP_MSG_PEER_DOWN_NOTIFICATION)
class BMPPeerDownNotification(BMPPeerMessage):
    """BMP Peer Down Notification Message

    ========================== ===============================================
    Attribute                  Description
    ========================== ===============================================
    version                    Version. this packet lib defines BMP ver. 3
    len                        Length field.  Ignored when encoding.
    type                       Type field.  one of BMP\\_MSG\\_ constants.
    reason                     Reason indicates why the session was closed.
    data                       vary by the reason.
    ========================== ===============================================
    """

    def __init__(self, reason, data, peer_type, is_post_policy, peer_distinguisher, peer_address, peer_as, peer_bgp_id, timestamp, version=VERSION, type_=BMP_MSG_PEER_DOWN_NOTIFICATION, len_=None, is_adj_rib_out=False):
        super(BMPPeerDownNotification, self).__init__(peer_type=peer_type, is_post_policy=is_post_policy, peer_distinguisher=peer_distinguisher, peer_address=peer_address, peer_as=peer_as, peer_bgp_id=peer_bgp_id, timestamp=timestamp, len_=len_, type_=type_, version=version, is_adj_rib_out=is_adj_rib_out)
        self.reason = reason
        self.data = data

    @classmethod
    def parser(cls, buf):
        kwargs, buf = super(BMPPeerDownNotification, cls).parser(buf)
        reason, = struct.unpack_from('!B', bytes(buf))
        buf = buf[struct.calcsize('!B'):]
        if reason == BMP_PEER_DOWN_REASON_LOCAL_BGP_NOTIFICATION:
            data, _, rest = BGPMessage.parser(buf)
        elif reason == BMP_PEER_DOWN_REASON_LOCAL_NO_NOTIFICATION:
            data = struct.unpack_from('!H', bytes(buf))
        elif reason == BMP_PEER_DOWN_REASON_REMOTE_BGP_NOTIFICATION:
            data, _, rest = BGPMessage.parser(buf)
        elif reason == BMP_PEER_DOWN_REASON_REMOTE_NO_NOTIFICATION:
            data = None
        else:
            reason = BMP_PEER_DOWN_REASON_UNKNOWN
            data = buf
        kwargs['reason'] = reason
        kwargs['data'] = data
        return kwargs

    def serialize_tail(self):
        msg = super(BMPPeerDownNotification, self).serialize_tail()
        msg += struct.pack('!B', self.reason)
        if self.reason == BMP_PEER_DOWN_REASON_LOCAL_BGP_NOTIFICATION:
            msg += self.data.serialize()
        elif self.reason == BMP_PEER_DOWN_REASON_LOCAL_NO_NOTIFICATION:
            msg += struct.pack('!H', self.data)
        elif self.reason == BMP_PEER_DOWN_REASON_REMOTE_BGP_NOTIFICATION:
            msg += self.data.serialize()
        elif self.reason == BMP_PEER_DOWN_REASON_UNKNOWN:
            msg += str(self.data)
        return msg