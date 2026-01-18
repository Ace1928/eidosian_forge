from yowsup.layers.protocol_acks.protocolentities.ack_outgoing import OutgoingAckProtocolEntity
from yowsup.structs.protocolentity import ProtocolEntityTest
import unittest
class OutgoingAckProtocolEntityTest(ProtocolEntityTest, unittest.TestCase):

    def setUp(self):
        self.ProtocolEntity = OutgoingAckProtocolEntity
        self.node = entity.toProtocolTreeNode()