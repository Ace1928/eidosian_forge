from unittest import mock
import uuid
from oslo_config import cfg
from troveclient import exceptions as troveexc
from troveclient.v1 import users
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import trove
from heat.engine import resource
from heat.engine.resources.openstack.trove import instance as dbinstance
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template as tmpl
from heat.tests import common
from heat.tests import utils
class InstanceTest(common.HeatTestCase):

    def setUp(self):
        super(InstanceTest, self).setUp()
        self.fc = mock.MagicMock()
        self.nova = mock.Mock()
        self.client = mock.Mock()
        self.patchobject(trove.TroveClientPlugin, '_create', return_value=self.client)
        self.stub_TroveFlavorConstraint_validate()
        self.patchobject(resource.Resource, 'is_using_neutron', return_value=True)
        self.flavor_resolve = self.patchobject(trove.TroveClientPlugin, 'find_flavor_by_name_or_id', return_value='1')
        self.fake_instance = FakeDBInstance()
        self.client.instances.create.return_value = self.fake_instance
        self.client.instances.get.return_value = self.fake_instance

    def _setup_test_instance(self, name, t, rsrc_name='MySqlCloudDB'):
        stack_name = '%s_stack' % name
        template = tmpl.Template(t)
        self.stack = parser.Stack(utils.dummy_context(), stack_name, template, stack_id=str(uuid.uuid4()))
        rsrc = self.stack[rsrc_name]
        rsrc.resource_id = '12345'
        return rsrc

    def _stubout_validate(self, instance, neutron=None, mock_net_constraint=False, with_port=True):
        if mock_net_constraint:
            self.stub_NetworkConstraint_validate()
        self.client.datastore_versions.list.return_value = [FakeVersion()]
        if neutron is not None:
            instance.is_using_neutron = mock.Mock(return_value=bool(neutron))
            if with_port:
                self.stub_PortConstraint_validate()

    def test_instance_create(self):
        t = template_format.parse(db_template)
        instance = self._setup_test_instance('dbinstance_create', t)
        scheduler.TaskRunner(instance.create)()
        self.assertEqual((instance.CREATE, instance.COMPLETE), instance.state)
        self.assertEqual('instances', instance.entity)

    def test_create_failed(self):
        t = template_format.parse(db_template)
        osdb_res = self._setup_test_instance('dbinstance_create', t)
        trove_mock = mock.Mock()
        self.patchobject(osdb_res, 'client', return_value=trove_mock)
        mock_input = mock.Mock()
        mock_input.status = 'ERROR'
        trove_mock.instances.get.return_value = mock_input
        error_string = 'Went to status ERROR due to "The last operation for the database instance failed due to an error."'
        exc = self.assertRaises(exception.ResourceInError, osdb_res.check_create_complete, mock_input)
        self.assertIn(error_string, str(exc))
        mock_input = mock.Mock()
        mock_input.status = 'FAILED'
        trove_mock.instances.get.return_value = mock_input
        error_string = 'Went to status FAILED due to "The database instance was created, but heat failed to set up the datastore. If a database instance is in the FAILED state, it should be deleted and a new one should be created."'
        exc = self.assertRaises(exception.ResourceInError, osdb_res.check_create_complete, mock_input)
        self.assertIn(error_string, str(exc))
        osdb_res.TROVE_STATUS_REASON = {}
        mock_input = mock.Mock()
        mock_input.status = 'ERROR'
        error_string = 'Went to status ERROR due to "Unknown"'
        trove_mock.instances.get.return_value = mock_input
        exc = self.assertRaises(exception.ResourceInError, osdb_res.check_create_complete, mock_input)
        self.assertIn(error_string, str(exc))

    def _create_failed_bad_status(self, status, error_message):
        t = template_format.parse(db_template)
        bad_instance = mock.Mock()
        bad_instance.status = status
        self.client.instances.get.return_value = bad_instance
        instance = self._setup_test_instance('test_bad_statuses', t)
        ex = self.assertRaises(exception.ResourceInError, instance.check_create_complete, self.fake_instance.id)
        self.assertIn(error_message, str(ex))

    def test_create_failed_status_error(self):
        self._create_failed_bad_status('ERROR', 'Went to status ERROR due to "The last operation for the database instance failed due to an error."')

    def test_create_failed_status_failed(self):
        self._create_failed_bad_status('FAILED', 'Went to status FAILED due to "The database instance was created, but heat failed to set up the datastore. If a database instance is in the FAILED state, it should be deleted and a new one should be created."')

    def test_instance_restore_point(self):
        t = template_format.parse(db_template)
        t['Resources']['MySqlCloudDB']['Properties']['restore_point'] = '1234'
        instance = self._setup_test_instance('dbinstance_create', t)
        self.client.flavors.get.side_effect = [troveexc.NotFound()]
        self.client.flavors.find.return_value = FakeFlavor(1, '1GB')
        scheduler.TaskRunner(instance.create)()
        self.assertEqual((instance.CREATE, instance.COMPLETE), instance.state)
        users = [{'name': 'testuser', 'password': 'pass', 'host': '%', 'databases': [{'name': 'validdb'}]}]
        databases = [{'collate': 'utf8_general_ci', 'character_set': 'utf8', 'name': 'validdb'}]
        self.client.instances.create.assert_called_once_with('test', '1', volume={'size': 30}, databases=databases, users=users, restorePoint={'backupRef': '1234'}, availability_zone=None, datastore='SomeDStype', datastore_version='MariaDB-5.5', nics=[], replica_of=None, replica_count=None)

    def test_instance_create_overlimit(self):
        t = template_format.parse(db_template)
        instance = self._setup_test_instance('dbinstance_create', t)
        self.client.instances.get.side_effect = [troveexc.RequestEntityTooLarge(), self.fake_instance]
        scheduler.TaskRunner(instance.create)()
        self.assertEqual((instance.CREATE, instance.COMPLETE), instance.state)

    def test_instance_create_fails(self):
        cfg.CONF.set_override('action_retry_limit', 0)
        t = template_format.parse(db_template)
        instance = self._setup_test_instance('dbinstance_create', t)
        self.fake_instance.status = 'ERROR'
        self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(instance.create))
        self.fake_instance.status = 'ACTIVE'

    def _get_db_instance(self):
        t = template_format.parse(db_template)
        res = self._setup_test_instance('trove_check', t)
        res.state_set(res.CREATE, res.COMPLETE)
        res.flavor = 'Foo Flavor'
        res.volume = 'Foo Volume'
        res.datastore_type = 'Foo Type'
        res.datastore_version = 'Foo Version'
        return res

    def test_instance_check(self):
        res = self._get_db_instance()
        scheduler.TaskRunner(res.check)()
        self.assertEqual((res.CHECK, res.COMPLETE), res.state)

    def test_instance_check_not_active(self):
        res = self._get_db_instance()
        self.fake_instance.status = 'FOOBAR'
        exc = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(res.check))
        self.assertIn('FOOBAR', str(exc))
        self.assertEqual((res.CHECK, res.FAILED), res.state)
        self.fake_instance.status = 'ACTIVE'

    def test_instance_delete(self):
        t = template_format.parse(db_template)
        instance = self._setup_test_instance('dbinstance_del', t)
        self.client.instances.get.side_effect = [self.fake_instance, troveexc.NotFound(404)]
        scheduler.TaskRunner(instance.create)()
        scheduler.TaskRunner(instance.delete)()

    def test_instance_delete_overlimit(self):
        t = template_format.parse(db_template)
        instance = self._setup_test_instance('dbinstance_del', t)
        self.client.instances.get.side_effect = [troveexc.RequestEntityTooLarge(), self.fake_instance, troveexc.NotFound(404)]
        scheduler.TaskRunner(instance.create)()
        scheduler.TaskRunner(instance.delete)()

    def test_instance_delete_resource_none(self):
        t = template_format.parse(db_template)
        instance = self._setup_test_instance('dbinstance_del', t)
        scheduler.TaskRunner(instance.create)()
        instance.resource_id = None
        scheduler.TaskRunner(instance.delete)()
        self.assertIsNone(instance.resource_id)

    def test_instance_resource_not_found(self):
        t = template_format.parse(db_template)
        instance = self._setup_test_instance('dbinstance_del', t)
        self.client.instances.get.side_effect = [self.fake_instance, troveexc.NotFound(404)]
        scheduler.TaskRunner(instance.create)()
        scheduler.TaskRunner(instance.delete)()

    def test_instance_attributes(self):
        fake_instance = FakeDBInstance()
        self.client.instances.create.return_value = fake_instance
        self.client.instances.get.return_value = fake_instance
        t = template_format.parse(db_template)
        instance = self._setup_test_instance('attr_test', t)
        self.assertEqual('testhost', instance.FnGetAtt('hostname'))
        self.assertEqual('https://adga23dd432a.rackspacecloud.com/132345245', instance.FnGetAtt('href'))

    def test_instance_validation_success(self):
        t = template_format.parse(db_template)
        instance = self._setup_test_instance('dbinstance_test', t)
        self._stubout_validate(instance)
        self.assertIsNone(instance.validate())

    def test_instance_validation_invalid_db(self):
        t = template_format.parse(db_template)
        t['Resources']['MySqlCloudDB']['Properties']['databases'] = [{'name': 'onedb'}]
        t['Resources']['MySqlCloudDB']['Properties']['users'] = [{'name': 'testuser', 'password': 'pass', 'databases': ['invaliddb']}]
        instance = self._setup_test_instance('dbinstance_test', t)
        self._stubout_validate(instance)
        ex = self.assertRaises(exception.StackValidationFailed, instance.validate)
        self.assertEqual("Database ['invaliddb'] specified for user does not exist in databases for resource MySqlCloudDB.", str(ex))

    def test_instance_validation_db_name_hyphens(self):
        t = template_format.parse(db_template)
        t['Resources']['MySqlCloudDB']['Properties']['databases'] = [{'name': '-foo-bar-'}]
        t['Resources']['MySqlCloudDB']['Properties']['users'] = [{'name': 'testuser', 'password': 'pass', 'databases': ['-foo-bar-']}]
        instance = self._setup_test_instance('dbinstance_test', t)
        self._stubout_validate(instance)
        self.assertIsNone(instance.validate())

    def test_instance_validation_users_none(self):
        t = template_format.parse(db_template)
        t['Resources']['MySqlCloudDB']['Properties']['users'] = []
        instance = self._setup_test_instance('dbinstance_test', t)
        self._stubout_validate(instance)
        self.assertIsNone(instance.validate())

    def test_instance_validation_databases_none(self):
        t = template_format.parse(db_template)
        t['Resources']['MySqlCloudDB']['Properties']['databases'] = []
        t['Resources']['MySqlCloudDB']['Properties']['users'] = [{'name': 'testuser', 'password': 'pass', 'databases': ['invaliddb']}]
        instance = self._setup_test_instance('dbinstance_test', t)
        self._stubout_validate(instance)
        ex = self.assertRaises(exception.StackValidationFailed, instance.validate)
        self.assertEqual('Databases property is required if users property is provided for resource MySqlCloudDB.', str(ex))

    def test_instance_validation_user_no_db(self):
        t = template_format.parse(db_template)
        t['Resources']['MySqlCloudDB']['Properties']['databases'] = [{'name': 'validdb'}]
        t['Resources']['MySqlCloudDB']['Properties']['users'] = [{'name': 'testuser', 'password': 'pass', 'databases': []}]
        instance = self._setup_test_instance('dbinstance_test', t)
        ex = self.assertRaises(exception.StackValidationFailed, instance.validate)
        self.assertEqual('Property error: Resources.MySqlCloudDB.Properties.users[0].databases: length (0) is out of range (min: 1, max: None)', str(ex))

    def test_instance_validation_no_datastore_yes_version(self):
        t = template_format.parse(db_template)
        t['Resources']['MySqlCloudDB']['Properties'].pop('datastore_type')
        instance = self._setup_test_instance('dbinstance_test', t)
        ex = self.assertRaises(exception.StackValidationFailed, instance.validate)
        exp_msg = 'Not allowed - datastore_version without datastore_type.'
        self.assertEqual(exp_msg, str(ex))

    def test_instance_validation_no_ds_version(self):
        t = template_format.parse(db_template)
        t['Resources']['MySqlCloudDB']['Properties']['datastore_type'] = 'mysql'
        t['Resources']['MySqlCloudDB']['Properties'].pop('datastore_version')
        instance = self._setup_test_instance('dbinstance_test', t)
        self._stubout_validate(instance)
        self.assertIsNone(instance.validate())

    def test_instance_validation_wrong_dsversion(self):
        t = template_format.parse(db_template)
        t['Resources']['MySqlCloudDB']['Properties']['datastore_type'] = 'mysql'
        t['Resources']['MySqlCloudDB']['Properties']['datastore_version'] = 'SomeVersion'
        instance = self._setup_test_instance('dbinstance_test', t)
        self._stubout_validate(instance)
        ex = self.assertRaises(exception.StackValidationFailed, instance.validate)
        expected_msg = 'Datastore version SomeVersion for datastore type mysql is not valid. Allowed versions are MariaDB-5.5.'
        self.assertEqual(expected_msg, str(ex))

    def test_instance_validation_implicit_version(self):
        t = template_format.parse(db_template)
        t['Resources']['MySqlCloudDB']['Properties']['datastore_type'] = 'mysql'
        t['Resources']['MySqlCloudDB']['Properties'].pop('datastore_version')
        instance = self._setup_test_instance('dbinstance_test', t)
        self.client.datastore_versions.list.return_value = [FakeVersion(), FakeVersion('MariaDB-5.0')]
        self.assertIsNone(instance.validate())

    def test_instance_validation_net_with_port_fail(self):
        t = template_format.parse(db_template)
        t['Resources']['MySqlCloudDB']['Properties']['networks'] = [{'port': 'someportuuid', 'network': 'somenetuuid'}]
        instance = self._setup_test_instance('dbinstance_test', t)
        self._stubout_validate(instance, neutron=True, mock_net_constraint=True)
        ex = self.assertRaises(exception.StackValidationFailed, instance.validate)
        self.assertEqual('Either network or port must be provided.', str(ex))

    def test_instance_validation_no_net_no_port_fail(self):
        t = template_format.parse(db_template)
        t['Resources']['MySqlCloudDB']['Properties']['networks'] = [{'fixed_ip': '1.2.3.4'}]
        instance = self._setup_test_instance('dbinstance_test', t)
        self._stubout_validate(instance, neutron=True, with_port=False)
        ex = self.assertRaises(exception.StackValidationFailed, instance.validate)
        self.assertEqual('Either network or port must be provided.', str(ex))

    def test_instance_validation_nic_port_on_novanet_fails(self):
        t = template_format.parse(db_template)
        t['Resources']['MySqlCloudDB']['Properties']['networks'] = [{'port': 'someportuuid'}]
        instance = self._setup_test_instance('dbinstance_test', t)
        self._stubout_validate(instance, neutron=False)
        ex = self.assertRaises(exception.StackValidationFailed, instance.validate)
        self.assertEqual('Can not use port property on Nova-network.', str(ex))

    def test_instance_create_with_port(self):
        t = template_format.parse(db_template_with_nics)
        instance = self._setup_test_instance('dbinstance_test', t)
        self.patchobject(neutron.NeutronClientPlugin, 'find_resourceid_by_name_or_id', return_value='someportid')
        self.stub_PortConstraint_validate()
        scheduler.TaskRunner(instance.create)()
        self.assertEqual((instance.CREATE, instance.COMPLETE), instance.state)
        self.client.instances.create.assert_called_once_with('test', '1', volume={'size': 30}, databases=[], users=[], restorePoint=None, availability_zone=None, datastore=None, datastore_version=None, nics=[{'port-id': 'someportid', 'v4-fixed-ip': '1.2.3.4'}], replica_of=None, replica_count=None)

    def test_instance_create_with_net_id(self):
        net_id = '034aa4d5-0f36-4127-8481-5caa5bfc9403'
        t = template_format.parse(db_template_with_nics)
        t['resources']['MySqlCloudDB']['properties']['networks'] = [{'network': net_id}]
        instance = self._setup_test_instance('dbinstance_test', t)
        self.stub_NetworkConstraint_validate()
        self.patchobject(neutron.NeutronClientPlugin, 'find_resourceid_by_name_or_id', return_value=net_id)
        scheduler.TaskRunner(instance.create)()
        self.assertEqual((instance.CREATE, instance.COMPLETE), instance.state)
        self.client.instances.create.assert_called_once_with('test', '1', volume={'size': 30}, databases=[], users=[], restorePoint=None, availability_zone=None, datastore=None, datastore_version=None, nics=[{'net-id': net_id}], replica_of=None, replica_count=None)

    def test_instance_create_with_replication(self):
        t = template_format.parse(db_template_with_replication)
        instance = self._setup_test_instance('dbinstance_test', t)
        scheduler.TaskRunner(instance.create)()
        self.assertEqual((instance.CREATE, instance.COMPLETE), instance.state)
        self.client.instances.create.assert_called_once_with('test', '1', volume={'size': 30}, databases=[], users=[], restorePoint=None, availability_zone=None, datastore=None, datastore_version=None, nics=[], replica_of='0e642916-dd64-43b3-933f-ff34fff69a7f', replica_count=2)

    def test_instance_get_live_state(self):
        self.fake_instance.to_dict = mock.Mock(return_value={'name': 'test_instance', 'flavor': {'id': '1'}, 'volume': {'size': 30}})
        fake_db1 = mock.Mock()
        fake_db1.name = 'validdb'
        fake_db2 = mock.Mock()
        fake_db2.name = 'secondvaliddb'
        self.client.databases.list.return_value = [fake_db1, fake_db2]
        expected = {'flavor': '1', 'name': 'test_instance', 'size': 30, 'databases': [{'name': 'validdb', 'character_set': 'utf8', 'collate': 'utf8_general_ci'}, {'name': 'secondvaliddb'}]}
        t = template_format.parse(db_template)
        instance = self._setup_test_instance('get_live_state_test', t)
        reality = instance.get_live_state(instance.properties)
        self.assertEqual(expected, reality)