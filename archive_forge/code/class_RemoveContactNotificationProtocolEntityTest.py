from yowsup.layers.protocol_contacts.protocolentities import RemoveContactNotificationProtocolEntity
from yowsup.structs.protocolentity import ProtocolEntityTest
import time
import unittest
class RemoveContactNotificationProtocolEntityTest(ProtocolEntityTest, unittest.TestCase):

    def setUp(self):
        self.ProtocolEntity = RemoveContactNotificationProtocolEntity
        self.node = entity.toProtocolTreeNode()