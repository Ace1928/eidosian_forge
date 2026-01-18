import operator
import os.path
import sys
import unittest
import testscenarios
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_protocol
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_3
from os_ken.ofproto import ofproto_v1_4
from os_ken.ofproto import ofproto_v1_5
from os_ken import exception
import json
class Test_Parser(testscenarios.WithScenarios, unittest.TestCase):
    """Test case for os_ken.ofproto, especially json representation"""
    scenarios = [(case['name'], case) for case in _list_test_cases()]

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @staticmethod
    def _msg_to_jsondict(msg):
        return msg.to_jsondict()

    @staticmethod
    def _jsondict_to_msg(dp, jsondict):
        return ofproto_parser.ofp_msg_from_jsondict(dp, jsondict)

    def test_parser(self):
        self._test_msg(name=self.name, wire_msg=self.wire_msg, json_str=self.json_str)

    def _test_msg(self, name, wire_msg, json_str):

        def bytes_eq(buf1, buf2):
            if buf1 != buf2:
                msg = 'EOF in either data'
                for i in range(0, min(len(buf1), len(buf2))):
                    c1 = operator.getitem(bytes(buf1), i)
                    c2 = operator.getitem(bytes(buf2), i)
                    if c1 != c2:
                        msg = 'differs at chr %d, %d != %d' % (i, c1, c2)
                        break
                assert buf1 == buf2, '%r != %r, %s' % (buf1, buf2, msg)
        json_dict = json.loads(json_str)
        version, msg_type, msg_len, xid = ofproto_parser.header(wire_msg)
        try:
            has_parser, has_serializer = implemented[version][msg_type]
        except KeyError:
            has_parser = True
            has_serializer = True
        dp = ofproto_protocol.ProtocolDesc(version=version)
        if has_parser:
            try:
                msg = ofproto_parser.msg(dp, version, msg_type, msg_len, xid, wire_msg)
                json_dict2 = self._msg_to_jsondict(msg)
            except exception.OFPTruncatedMessage as e:
                json_dict2 = {'OFPTruncatedMessage': self._msg_to_jsondict(e.ofpmsg)}
            with open('/tmp/%s.json' % name, 'w') as _file:
                _file.write(json.dumps(json_dict2))
            self.assertEqual(json_dict, json_dict2)
            if 'OFPTruncatedMessage' in json_dict2:
                return
        xid = json_dict[list(json_dict.keys())[0]].pop('xid', None)
        msg2 = self._jsondict_to_msg(dp, json_dict)
        msg2.set_xid(xid)
        if has_serializer:
            msg2.serialize()
            self.assertEqual(self._msg_to_jsondict(msg2), json_dict)
            bytes_eq(wire_msg, msg2.buf)

            def _remove(d, names):
                f = lambda x: _remove(x, names)
                if isinstance(d, list):
                    return list(map(f, d))
                if isinstance(d, dict):
                    d2 = {}
                    for k, v in d.items():
                        if k in names:
                            continue
                        d2[k] = f(v)
                    return d2
                return d
            json_dict3 = _remove(json_dict, ['len', 'length'])
            msg3 = self._jsondict_to_msg(dp, json_dict3)
            msg3.set_xid(xid)
            msg3.serialize()
            bytes_eq(wire_msg, msg3.buf)
            msg2.serialize()
            bytes_eq(wire_msg, msg2.buf)