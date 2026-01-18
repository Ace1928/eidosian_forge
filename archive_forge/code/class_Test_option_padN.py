import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib.packet import ipv6
class Test_option_padN(Test_option):

    def setUp(self):
        self.type_ = 1
        self.len_ = 0
        self.data = None
        self.opt = ipv6.option(self.type_, self.len_, self.data)
        self.form = '!BB'
        self.buf = struct.pack(self.form, self.type_, self.len_)

    def test_serialize(self):
        buf = self.opt.serialize()
        res = struct.unpack_from(self.form, buf)
        self.assertEqual(self.type_, res[0])
        self.assertEqual(self.len_, res[1])