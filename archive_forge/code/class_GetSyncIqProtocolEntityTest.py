from yowsup.layers.protocol_contacts.protocolentities.iq_sync_get import GetSyncIqProtocolEntity
from yowsup.structs.protocolentity import ProtocolEntityTest
import unittest
class GetSyncIqProtocolEntityTest(ProtocolEntityTest, unittest.TestCase):

    def setUp(self):
        self.ProtocolEntity = GetSyncIqProtocolEntity
        self.node = entity.toProtocolTreeNode()