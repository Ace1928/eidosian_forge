import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib.packet import ipv6
class Test_option(unittest.TestCase):

    def setUp(self):
        self.type_ = 5
        self.data = b'\x00\x00'
        self.len_ = len(self.data)
        self.opt = ipv6.option(self.type_, self.len_, self.data)
        self.form = '!BB%ds' % self.len_
        self.buf = struct.pack(self.form, self.type_, self.len_, self.data)

    def tearDown(self):
        pass

    def test_init(self):
        self.assertEqual(self.type_, self.opt.type_)
        self.assertEqual(self.len_, self.opt.len_)
        self.assertEqual(self.data, self.opt.data)

    def test_parser(self):
        _res = ipv6.option.parser(self.buf)
        if type(_res) is tuple:
            res = _res[0]
        else:
            res = _res
        self.assertEqual(self.type_, res.type_)
        self.assertEqual(self.len_, res.len_)
        self.assertEqual(self.data, res.data)

    def test_serialize(self):
        buf = self.opt.serialize()
        res = struct.unpack_from(self.form, buf)
        self.assertEqual(self.type_, res[0])
        self.assertEqual(self.len_, res[1])
        self.assertEqual(self.data, res[2])

    def test_len(self):
        self.assertEqual(len(self.opt), 2 + self.len_)