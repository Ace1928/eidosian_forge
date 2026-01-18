from io import BytesIO
from dulwich.tests import TestCase
from ..errors import HangupException
from ..protocol import (
class ProtocolTests(BaseProtocolTests, TestCase):

    def setUp(self):
        TestCase.setUp(self)
        self.rout = BytesIO()
        self.rin = BytesIO()
        self.proto = Protocol(self.rin.read, self.rout.write)