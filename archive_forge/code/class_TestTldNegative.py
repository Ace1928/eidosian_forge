from tempest.lib.exceptions import CommandFailed
from designateclient.functionaltests.base import BaseDesignateTest
from designateclient.functionaltests.datagen import random_tld
from designateclient.functionaltests.v2.fixtures import TLDFixture
class TestTldNegative(BaseDesignateTest):

    def test_tld_invalid_commmand(self):
        client = self.clients.as_user('admin')
        self.assertRaises(CommandFailed, client.openstack, 'tld notacommand')

    def test_tld_create_invalid_flag(self):
        client = self.clients.as_user('admin')
        self.assertRaises(CommandFailed, client.openstack, 'tld create --notanoption "junk"')