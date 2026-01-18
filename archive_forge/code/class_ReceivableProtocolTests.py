from io import BytesIO
from dulwich.tests import TestCase
from ..errors import HangupException
from ..protocol import (
class ReceivableProtocolTests(BaseProtocolTests, TestCase):

    def setUp(self):
        TestCase.setUp(self)
        self.rout = BytesIO()
        self.rin = ReceivableBytesIO()
        self.proto = ReceivableProtocol(self.rin.recv, self.rout.write)
        self.proto._rbufsize = 8

    def test_eof(self):
        self.rin.allow_read_past_eof = True
        BaseProtocolTests.test_eof(self)

    def test_recv(self):
        all_data = b'1234567' * 10
        self.rin.write(all_data)
        self.rin.seek(0)
        data = b''
        for _ in range(10):
            data += self.proto.recv(10)
        self.assertRaises(GitProtocolError, self.proto.recv, 10)
        self.assertEqual(all_data, data)

    def test_recv_read(self):
        all_data = b'1234567'
        self.rin.write(all_data)
        self.rin.seek(0)
        self.assertEqual(b'1234', self.proto.recv(4))
        self.assertEqual(b'567', self.proto.read(3))
        self.assertRaises(GitProtocolError, self.proto.recv, 10)

    def test_read_recv(self):
        all_data = b'12345678abcdefg'
        self.rin.write(all_data)
        self.rin.seek(0)
        self.assertEqual(b'1234', self.proto.read(4))
        self.assertEqual(b'5678abc', self.proto.recv(8))
        self.assertEqual(b'defg', self.proto.read(4))
        self.assertRaises(GitProtocolError, self.proto.recv, 10)

    def test_mixed(self):
        all_data = b','.join((str(i).encode('ascii') for i in range(100)))
        self.rin.write(all_data)
        self.rin.seek(0)
        data = b''
        for i in range(1, 100):
            data += self.proto.recv(i)
            if len(data) + i > len(all_data):
                data += self.proto.recv(i)
                data += self.proto.recv(1)
                break
            else:
                data += self.proto.read(i)
        else:
            self.fail()
        self.assertEqual(all_data, data)