from tempest.lib.exceptions import CommandFailed
from designateclient.functionaltests.base import BaseDesignateTest
from designateclient.functionaltests.datagen import random_a_recordset_name
from designateclient.functionaltests.datagen import random_zone_name
from designateclient.functionaltests.v2.fixtures import RecordsetFixture
from designateclient.functionaltests.v2.fixtures import ZoneFixture
class TestRecordsetNegative(BaseDesignateTest):

    def test_invalid_option_on_recordset_create(self):
        cmd = 'recordset create de47d30b-41c5-4e38-b2c5-e0b908e19ec7 aaa.desig.com. --type A --record 1.2.3.4 --invalid "not valid"'
        self.assertRaises(CommandFailed, self.clients.openstack, cmd)

    def test_invalid_recordset_command(self):
        cmd = 'recordset hopefullynotvalid'
        self.assertRaises(CommandFailed, self.clients.openstack, cmd)