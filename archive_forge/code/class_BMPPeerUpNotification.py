import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import packet_base
from os_ken.lib.packet import stream_parser
from os_ken.lib.packet.bgp import BGPMessage
from os_ken.lib.type_desc import TypeDisp
@BMPMessage.register_type(BMP_MSG_PEER_UP_NOTIFICATION)
class BMPPeerUpNotification(BMPPeerMessage):
    """BMP Peer Up Notification Message

    ========================== ===============================================
    Attribute                  Description
    ========================== ===============================================
    version                    Version. this packet lib defines BMP ver. 3
    len                        Length field.  Ignored when encoding.
    type                       Type field.  one of BMP\\_MSG\\_ constants.
    peer_type                  The type of the peer.
    peer_flags                 Provide more information about the peer.
    peer_distinguisher         Use for L3VPN router which can have multiple
                               instance.
    peer_address               The remote IP address associated with the TCP
                               session.
    peer_as                    The Autonomous System number of the peer.
    peer_bgp_id                The BGP Identifier of the peer
    timestamp                  The time when the encapsulated routes were
                               received.
    local_address              The local IP address associated with the
                               peering TCP session.
    local_port                 The local port number associated with the
                               peering TCP session.
    remote_port                The remote port number associated with the
                               peering TCP session.
    sent_open_message          The full OPEN message transmitted by the
                               monitored router to its peer.
    received_open_message      The full OPEN message received by the monitored
                               router from its peer.
    ========================== ===============================================
    """
    _PACK_STR = '!16sHH'
    _MIN_LEN = struct.calcsize(_PACK_STR)

    def __init__(self, local_address, local_port, remote_port, sent_open_message, received_open_message, peer_type, is_post_policy, peer_distinguisher, peer_address, peer_as, peer_bgp_id, timestamp, version=VERSION, type_=BMP_MSG_PEER_UP_NOTIFICATION, len_=None, is_adj_rib_out=False):
        super(BMPPeerUpNotification, self).__init__(peer_type=peer_type, is_post_policy=is_post_policy, peer_distinguisher=peer_distinguisher, peer_address=peer_address, peer_as=peer_as, peer_bgp_id=peer_bgp_id, timestamp=timestamp, len_=len_, type_=type_, version=version, is_adj_rib_out=is_adj_rib_out)
        self.local_address = local_address
        self.local_port = local_port
        self.remote_port = remote_port
        self.sent_open_message = sent_open_message
        self.received_open_message = received_open_message

    @classmethod
    def parser(cls, buf):
        kwargs, rest = super(BMPPeerUpNotification, cls).parser(buf)
        local_address, local_port, remote_port = struct.unpack_from(cls._PACK_STR, bytes(rest))
        if '.' in kwargs['peer_address']:
            local_address = addrconv.ipv4.bin_to_text(local_address[-4:])
        elif ':' in kwargs['peer_address']:
            local_address = addrconv.ipv6.bin_to_text(local_address)
        else:
            raise ValueError('invalid local_address: %s' % local_address)
        kwargs['local_address'] = local_address
        kwargs['local_port'] = local_port
        kwargs['remote_port'] = remote_port
        rest = rest[cls._MIN_LEN:]
        sent_open_msg, _, rest = BGPMessage.parser(rest)
        received_open_msg, _, rest = BGPMessage.parser(rest)
        kwargs['sent_open_message'] = sent_open_msg
        kwargs['received_open_message'] = received_open_msg
        return kwargs

    def serialize_tail(self):
        msg = super(BMPPeerUpNotification, self).serialize_tail()
        if '.' in self.local_address:
            local_address = struct.pack('!12x4s', addrconv.ipv4.text_to_bin(self.local_address))
        elif ':' in self.local_address:
            local_address = addrconv.ipv6.text_to_bin(self.local_address)
        else:
            raise ValueError('invalid local_address: %s' % self.local_address)
        msg += struct.pack(self._PACK_STR, local_address, self.local_port, self.remote_port)
        msg += self.sent_open_message.serialize()
        msg += self.received_open_message.serialize()
        return msg