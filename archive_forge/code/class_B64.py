from functools import reduce
import unittest
import testscenarios
from os_ken.ofproto import ofproto_v1_5
from os_ken.ofproto import ofproto_v1_5_parser
class B64(Field):

    @staticmethod
    def generate():
        yield 'aG9nZWhvZ2U='
        yield 'ZnVnYWZ1Z2E='