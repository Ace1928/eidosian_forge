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
@_register_exp_type(ofproto_common.ONF_EXPERIMENTER_ID, ofproto.ONF_ET_BUNDLE_CONTROL)
class ONFBundleCtrlMsg(OFPExperimenter):
    """
    Bundle control message

    The controller uses this message to create, destroy and commit bundles

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    bundle_id        Id of the bundle
    type             One of the following values.

                     | ONF_BCT_OPEN_REQUEST
                     | ONF_BCT_OPEN_REPLY
                     | ONF_BCT_CLOSE_REQUEST
                     | ONF_BCT_CLOSE_REPLY
                     | ONF_BCT_COMMIT_REQUEST
                     | ONF_BCT_COMMIT_REPLY
                     | ONF_BCT_DISCARD_REQUEST
                     | ONF_BCT_DISCARD_REPLY
    flags            Bitmap of the following flags.

                     | ONF_BF_ATOMIC
                     | ONF_BF_ORDERED
    properties       List of ``OFPBundleProp`` subclass instance
    ================ ======================================================

    Example::

        def send_bundle_control(self, datapath):
            ofp = datapath.ofproto
            ofp_parser = datapath.ofproto_parser

            req = ofp_parser.ONFBundleCtrlMsg(datapath, 7,
                                              ofp.ONF_BCT_OPEN_REQUEST,
                                              ofp.ONF_BF_ATOMIC, [])
            datapath.send_msg(req)
    """

    def __init__(self, datapath, bundle_id=None, type_=None, flags=None, properties=None):
        super(ONFBundleCtrlMsg, self).__init__(datapath, ofproto_common.ONF_EXPERIMENTER_ID, ofproto.ONF_ET_BUNDLE_CONTROL)
        self.bundle_id = bundle_id
        self.type = type_
        self.flags = flags
        self.properties = properties

    def _serialize_body(self):
        bin_props = bytearray()
        for p in self.properties:
            bin_props += p.serialize()
        msg_pack_into(ofproto.OFP_EXPERIMENTER_HEADER_PACK_STR, self.buf, ofproto.OFP_HEADER_SIZE, self.experimenter, self.exp_type)
        msg_pack_into(ofproto.ONF_BUNDLE_CTRL_PACK_STR, self.buf, ofproto.OFP_EXPERIMENTER_HEADER_SIZE, self.bundle_id, self.type, self.flags)
        self.buf += bin_props

    @classmethod
    def parser_subtype(cls, super_msg):
        bundle_id, type_, flags = struct.unpack_from(ofproto.ONF_BUNDLE_CTRL_PACK_STR, super_msg.data)
        msg = cls(super_msg.datapath, bundle_id, type_, flags)
        msg.properties = []
        rest = super_msg.data[ofproto.ONF_BUNDLE_CTRL_SIZE:]
        while rest:
            p, rest = OFPBundleProp.parse(rest)
            msg.properties.append(p)
        return msg