import unittest
import logging
from os_ken.lib.packet import llc
class Test_ControlFormatI(unittest.TestCase):
    msg = llc.llc(llc.SAP_BPDU, llc.SAP_BPDU, llc.ControlFormatI())

    def test_json(self):
        jsondict = self.msg.to_jsondict()
        msg = llc.llc.from_jsondict(jsondict['llc'])
        self.assertEqual(str(self.msg), str(msg))