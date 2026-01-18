from functools import reduce
import unittest
import testscenarios
from os_ken.ofproto import ofproto_v1_5
from os_ken.ofproto import ofproto_v1_5_parser
class Int4(Field):

    @staticmethod
    def generate():
        yield 0
        yield 305419896
        yield 4294967295