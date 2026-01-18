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
@_set_msg_type(ofproto.OFPT_PACKET_OUT)
class OFPPacketOut(MsgBase):
    """
    Packet-Out message

    The controller uses this message to send a packet out throught the
    switch.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    buffer_id        ID assigned by datapath (OFP_NO_BUFFER if none)
    in_port          Packet's input port or ``OFPP_CONTROLLER``
    actions          list of OpenFlow action class
    data             Packet data of a binary type value or
                     an instances of packet.Packet.
    ================ ======================================================

    Example::

        def send_packet_out(self, datapath, buffer_id, in_port):
            ofp = datapath.ofproto
            ofp_parser = datapath.ofproto_parser

            actions = [ofp_parser.OFPActionOutput(ofp.OFPP_FLOOD, 0)]
            req = ofp_parser.OFPPacketOut(datapath, buffer_id,
                                          in_port, actions)
            datapath.send_msg(req)
    """

    def __init__(self, datapath, buffer_id=None, in_port=None, actions=None, data=None, actions_len=None):
        assert in_port is not None
        super(OFPPacketOut, self).__init__(datapath)
        self.buffer_id = buffer_id
        self.in_port = in_port
        self.actions_len = 0
        self.actions = actions
        self.data = data

    def _serialize_body(self):
        self.actions_len = 0
        offset = ofproto.OFP_PACKET_OUT_SIZE
        for a in self.actions:
            a.serialize(self.buf, offset)
            offset += a.len
            self.actions_len += a.len
        if self.data is not None:
            assert self.buffer_id == 4294967295
            if isinstance(self.data, packet.Packet):
                self.data.serialize()
                self.buf += self.data.data
            else:
                self.buf += self.data
        msg_pack_into(ofproto.OFP_PACKET_OUT_PACK_STR, self.buf, ofproto.OFP_HEADER_SIZE, self.buffer_id, self.in_port, self.actions_len)

    @classmethod
    def from_jsondict(cls, dict_, decode_string=base64.b64decode, **additional_args):
        if isinstance(dict_['data'], dict):
            data = dict_.pop('data')
            ins = super(OFPPacketOut, cls).from_jsondict(dict_, decode_string, **additional_args)
            ins.data = packet.Packet.from_jsondict(data['Packet'])
            dict_['data'] = data
        else:
            ins = super(OFPPacketOut, cls).from_jsondict(dict_, decode_string, **additional_args)
        return ins