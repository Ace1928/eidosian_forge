from functools import reduce
import unittest
import testscenarios
from os_ken.ofproto import ofproto_v1_5
from os_ken.ofproto import ofproto_v1_5_parser
class Int4double(Field):

    @staticmethod
    def generate():
        yield [0, 1]
        yield [305419896, 591751049]
        yield [4294967295, 4294967294]