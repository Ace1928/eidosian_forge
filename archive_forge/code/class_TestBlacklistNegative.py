from tempest.lib.exceptions import CommandFailed
from designateclient.functionaltests.base import BaseDesignateTest
from designateclient.functionaltests.datagen import random_blacklist
from designateclient.functionaltests.v2.fixtures import BlacklistFixture
class TestBlacklistNegative(BaseDesignateTest):

    def test_invalid_blacklist_command(self):
        client = self.clients.as_user('admin')
        cmd = 'zone blacklist notacommand'
        self.assertRaises(CommandFailed, client.openstack, cmd)

    def test_blacklist_create_invalid_flag(self):
        client = self.clients.as_user('admin')
        cmd = 'zone blacklist create --pattern helloworld --notaflag invalid'
        self.assertRaises(CommandFailed, client.openstack, cmd)