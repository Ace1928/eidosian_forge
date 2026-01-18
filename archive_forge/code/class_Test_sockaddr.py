import unittest
import platform
import sys
from os_ken.lib import sockaddr
class Test_sockaddr(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_sockaddr_linux_sa_in4(self):
        if system != 'Linux' or sys.byteorder != 'little':
            return
        addr = '127.0.0.1'
        expected_result = b'\x02\x00\x00\x00\x7f\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00'
        self.assertEqual(expected_result, sockaddr.sa_in4(addr))

    def test_sockaddr_linux_sa_in6(self):
        if system != 'Linux' or sys.byteorder != 'little':
            return
        addr = 'dead:beef::1'
        expected_result = b'\n\x00\x00\x00\x00\x00\x00\x00\xde\xad\xbe\xef\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00'
        self.assertEqual(expected_result, sockaddr.sa_in6(addr))

    def test_sockaddr_sa_to_ss(self):
        addr = b'\x01'
        expected_result = b'\x01' + 127 * b'\x00'
        self.assertEqual(expected_result, sockaddr.sa_to_ss(addr))