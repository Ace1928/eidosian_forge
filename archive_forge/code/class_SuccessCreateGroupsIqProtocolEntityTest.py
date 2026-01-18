from yowsup.structs.protocolentity import ProtocolEntityTest
from yowsup.layers.protocol_groups.protocolentities import SuccessCreateGroupsIqProtocolEntity
import unittest
class SuccessCreateGroupsIqProtocolEntityTest(ProtocolEntityTest, unittest.TestCase):

    def setUp(self):
        self.ProtocolEntity = SuccessCreateGroupsIqProtocolEntity
        self.node = entity.toProtocolTreeNode()