import unittest
import logging
from os_ken.ofproto.ofproto_v1_0_parser import *
from os_ken.ofproto.nx_actions import *
from os_ken.ofproto import ofproto_v1_0_parser
from os_ken.lib import addrconv
class TestOFPPortStatsReply(unittest.TestCase):
    """ Test case for ofproto_v1_0_parser.OFPPortStatsReply
    """

    class Datapath(object):
        ofproto = ofproto
        ofproto_parser = ofproto_v1_0_parser
    c = OFPPortStatsReply(Datapath)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        pass

    def test_parser(self):
        version = {'buf': b'\x01', 'val': ofproto.OFP_VERSION}
        msg_type = {'buf': b'\x11', 'val': ofproto.OFPT_STATS_REPLY}
        msg_len_val = ofproto.OFP_STATS_MSG_SIZE + ofproto.OFP_PORT_STATS_SIZE
        msg_len = {'buf': b'\x00t', 'val': msg_len_val}
        xid = {'buf': b'\xc2\xaf=\xff', 'val': 3266264575}
        buf = version['buf'] + msg_type['buf'] + msg_len['buf'] + xid['buf']
        type_ = {'buf': b'\x00\x04', 'val': ofproto.OFPST_PORT}
        flags = {'buf': b'\xda\xde', 'val': 56030}
        buf += type_['buf'] + flags['buf']
        port_no = {'buf': b'\xe7k', 'val': 59243}
        zfill = b'\x00' * 6
        rx_packets = {'buf': b'SD6a\xc4\x86\xc07', 'val': 5999980397101236279}
        tx_packets = {'buf': b"'\xa4A\xd7\xd4S\x9eB", 'val': 2856480458895760962}
        rx_bytes = {'buf': b'U\xa18`C\x97\r\x89', 'val': 6170274950576278921}
        tx_bytes = {'buf': b'w\xe1\xd5c\x18\xaec\xaa', 'val': 8638420181865882538}
        rx_dropped = {'buf': b'`\xe6 \x01$\xdaNZ', 'val': 6982303461569875546}
        tx_dropped = {'buf': b'\t-]qq\xb6\x8e\xc7', 'val': 661287462113808071}
        rx_errors = {'buf': b'/~5\xb3f<\x19\r', 'val': 3422231811478788365}
        tx_errors = {'buf': b'W2\x08/\x882@k', 'val': 6283093430376743019}
        rx_frame_err = {'buf': b'\x0c(o\xad\xcefn\x8b', 'val': 876072919806406283}
        rx_over_err = {'buf': b'Z\x90\x8f\x9b\xfc\x82.\xa0', 'val': 6525873760178941600}
        rx_crc_err = {'buf': b's:q\x17\xd6tiG', 'val': 8303073210207070535}
        collisions = {'buf': b'/R\x0cy\x96\x03ny', 'val': 3409801584220270201}
        buf += port_no['buf'] + zfill + rx_packets['buf'] + tx_packets['buf'] + rx_bytes['buf'] + tx_bytes['buf'] + rx_dropped['buf'] + tx_dropped['buf'] + rx_errors['buf'] + tx_errors['buf'] + rx_frame_err['buf'] + rx_over_err['buf'] + rx_crc_err['buf'] + collisions['buf']
        res = OFPPortStatsReply.parser(object, version['val'], msg_type['val'], msg_len['val'], xid['val'], buf)
        self.assertEqual(version['val'], res.version)
        self.assertEqual(msg_type['val'], res.msg_type)
        self.assertEqual(msg_len['val'], res.msg_len)
        self.assertEqual(xid['val'], res.xid)
        self.assertEqual(type_['val'], res.type)
        self.assertEqual(flags['val'], res.flags)
        body = res.body[0]
        self.assertEqual(port_no['val'], body.port_no)
        self.assertEqual(rx_packets['val'], body.rx_packets)
        self.assertEqual(tx_packets['val'], body.tx_packets)
        self.assertEqual(rx_bytes['val'], body.rx_bytes)
        self.assertEqual(tx_bytes['val'], body.tx_bytes)
        self.assertEqual(rx_dropped['val'], body.rx_dropped)
        self.assertEqual(tx_dropped['val'], body.tx_dropped)
        self.assertEqual(rx_errors['val'], body.rx_errors)
        self.assertEqual(tx_errors['val'], body.tx_errors)
        self.assertEqual(rx_frame_err['val'], body.rx_frame_err)
        self.assertEqual(rx_over_err['val'], body.rx_over_err)
        self.assertEqual(rx_crc_err['val'], body.rx_crc_err)
        self.assertEqual(collisions['val'], body.collisions)

    def test_serialize(self):
        pass