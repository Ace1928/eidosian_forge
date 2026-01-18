from yowsup.layers.protocol_contacts.protocolentities import AddContactNotificationProtocolEntity
from yowsup.structs.protocolentity import ProtocolEntityTest
import time
import unittest
class AddContactNotificationProtocolEntityTest(ProtocolEntityTest, unittest.TestCase):

    def setUp(self):
        super(AddContactNotificationProtocolEntityTest, self).setUp()
        self.ProtocolEntity = AddContactNotificationProtocolEntity
        self.node = entity.toProtocolTreeNode()