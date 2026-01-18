from unittest import mock
from osc_lib import utils
from oslo_utils import uuidutils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import database_instances
from troveclient.tests.osc.v1 import fakes
from troveclient.v1 import instances
class TestDatabaseInstanceCreate(TestInstances):
    values = ('2017-12-22T20:02:32', 'mysql', '5.6', '5.7.29', '310', '2468', 'test', 'test-net', 'net-id', 'BUILD', '2017-12-22T20:02:32', 1)
    columns = ('created', 'datastore', 'datastore_version', 'datastore_version_number', 'flavor', 'id', 'name', 'networks', 'networks_id', 'status', 'updated', 'volume')

    def setUp(self):
        super(TestDatabaseInstanceCreate, self).setUp()
        self.cmd = database_instances.CreateDatabaseInstance(self.app, None)
        self.data = self.fake_instances.get_instance_create()
        self.instance_client.create.return_value = self.data

    @mock.patch.object(utils, 'find_resource')
    def test_instance_create(self, mock_find):
        args = ['test-name', '--flavor', '103', '--size', '1', '--databases', 'db1', 'db2', '--users', 'u1:111', 'u2:111', '--datastore', 'datastore', '--datastore-version', 'datastore_version', '--nic', 'net-id=net1', '--replica-of', 'test', '--replica-count', '4', '--module', 'mod_id', '--is-public', '--allowed-cidr', '10.0.0.1/24', '--allowed-cidr', '192.168.0.1/24']
        verifylist = [('name', 'test-name'), ('flavor', '103'), ('size', 1), ('databases', ['db1', 'db2']), ('users', ['u1:111', 'u2:111']), ('datastore', 'datastore'), ('datastore_version', 'datastore_version'), ('nics', 'net-id=net1'), ('replica_of', 'test'), ('replica_count', 4), ('modules', ['mod_id']), ('is_public', True), ('allowed_cidrs', ['10.0.0.1/24', '192.168.0.1/24'])]
        parsed_args = self.check_parser(self.cmd, args, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.values, data)

    @mock.patch.object(utils, 'find_resource')
    def test_instance_create_without_allowed_cidrs(self, mock_find):
        resp = {'id': 'a1fea1cf-18ad-48ab-bdfd-fce99a4b834e', 'name': 'test-mysql', 'status': 'BUILD', 'flavor': {'id': 'a48ea749-7ee3-4003-8aae-eb4e79773e2d'}, 'datastore': {'type': 'mysql', 'version': '5.7.29', 'version_number': '5.7.29'}, 'region': 'RegionOne', 'access': {'is_public': True}, 'volume': {'size': 1}, 'created': '2020-08-12T09:41:47', 'updated': '2020-08-12T09:41:47', 'service_status_updated': '2020-08-12T09:41:47'}
        self.instance_client.create.return_value = instances.Instance(mock.MagicMock(), resp)
        args = ['test-mysql', '--flavor', 'a48ea749-7ee3-4003-8aae-eb4e79773e2d', '--size', '1', '--datastore', 'mysql', '--datastore-version', '5.7.29', '--nic', 'net-id=net1', '--is-public']
        verifylist = [('name', 'test-mysql'), ('flavor', 'a48ea749-7ee3-4003-8aae-eb4e79773e2d'), ('size', 1), ('datastore', 'mysql'), ('datastore_version', '5.7.29'), ('nics', 'net-id=net1'), ('is_public', True), ('allowed_cidrs', None)]
        parsed_args = self.check_parser(self.cmd, args, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        expected_columns = ('allowed_cidrs', 'created', 'datastore', 'datastore_version', 'datastore_version_number', 'flavor', 'id', 'name', 'public', 'region', 'service_status_updated', 'status', 'updated', 'volume')
        expected_values = ([], '2020-08-12T09:41:47', 'mysql', '5.7.29', '5.7.29', 'a48ea749-7ee3-4003-8aae-eb4e79773e2d', 'a1fea1cf-18ad-48ab-bdfd-fce99a4b834e', 'test-mysql', True, 'RegionOne', '2020-08-12T09:41:47', 'BUILD', '2020-08-12T09:41:47', 1)
        self.assertEqual(expected_columns, columns)
        self.assertEqual(expected_values, data)

    @mock.patch.object(utils, 'find_resource')
    def test_instance_create_nic_param(self, mock_find):
        fake_id = self.random_uuid()
        mock_find.return_value.id = fake_id
        args = ['test-mysql', '--flavor', 'a48ea749-7ee3-4003-8aae-eb4e79773e2d', '--size', '1', '--datastore', 'mysql', '--datastore-version', '5.7.29', '--nic', 'net-id=net1,subnet-id=subnet_id,ip-address=192.168.1.11']
        parsed_args = self.check_parser(self.cmd, args, [])
        self.cmd.take_action(parsed_args)
        self.instance_client.create.assert_called_once_with('test-mysql', flavor_id=fake_id, volume={'size': 1, 'type': None}, databases=[], users=[], restorePoint=None, availability_zone=None, datastore='mysql', datastore_version='5.7.29', datastore_version_number=None, nics=[{'network_id': 'net1', 'subnet_id': 'subnet_id', 'ip_address': '192.168.1.11'}], configuration=None, replica_of=None, replica_count=None, modules=[], locality=None, region_name=None, access={'is_public': False})