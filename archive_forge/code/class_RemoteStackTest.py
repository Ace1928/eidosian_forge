import collections
import json
from unittest import mock
from heatclient import exc
from heatclient.v1 import stacks
from keystoneauth1 import loading as ks_loading
from oslo_config import cfg
from heat.common import exception
from heat.common.i18n import _
from heat.common import policy
from heat.common import template_format
from heat.engine.clients.os import heat_plugin
from heat.engine import environment
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.openstack.heat import remote_stack
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack
from heat.engine import template
from heat.tests import common as tests_common
from heat.tests import utils
class RemoteStackTest(tests_common.HeatTestCase):

    def setUp(self):
        super(RemoteStackTest, self).setUp()
        self.this_region = 'RegionOne'
        self.that_region = 'RegionTwo'
        self.bad_region = 'RegionNone'
        cfg.CONF.set_override('action_retry_limit', 0)
        self.parent = None
        self.heat = None
        self.client_plugin = None
        self.this_context = None
        self.old_clients = None

        def unset_clients_property():
            if self.this_context is not None:
                type(self.this_context).clients = self.old_clients
        self.addCleanup(unset_clients_property)

    def initialize(self, stack_template=None):
        parent, rsrc = self.create_parent_stack(remote_region='RegionTwo', stack_template=stack_template)
        self.parent = parent
        self.heat = rsrc._context().clients.client('heat')
        self.client_plugin = rsrc._context().clients.client_plugin('heat')

    def create_parent_stack(self, remote_region=None, custom_template=None, stack_template=None):
        if not stack_template:
            stack_template = parent_stack_template
        snippet = template_format.parse(stack_template)
        self.files = {'remote_template.yaml': custom_template or remote_template}
        region_name = remote_region or self.this_region
        props = snippet['resources']['remote_stack']['properties']
        if remote_region is None:
            del props['context']
        else:
            props['context']['region_name'] = region_name
        if self.this_context is None:
            self.this_context = utils.dummy_context(region_name=self.this_region)
        tmpl = template.Template(snippet, files=self.files)
        parent = stack.Stack(self.this_context, 'parent_stack', tmpl)
        ctx = parent.context.to_dict()
        self.assertEqual(self.this_region, ctx['region_name'])
        self.assertEqual(self.this_context.to_dict(), ctx)
        parent.store()
        resource_defns = parent.t.resource_definitions(parent)
        rsrc = remote_stack.RemoteStack('remote_stack_res', resource_defns['remote_stack'], parent)
        self.assertEqual(60, rsrc.properties.get('timeout'))
        remote_context = rsrc._context()
        hc = FakeClients(self.this_context, rsrc._region_name)
        if self.old_clients is None:
            self.old_clients = type(remote_context).clients
            type(remote_context).clients = mock.PropertyMock(return_value=hc)
        return (parent, rsrc)

    def create_remote_stack(self, stack_template=None):
        defaults = [get_stack(stack_status='CREATE_IN_PROGRESS'), get_stack(stack_status='CREATE_COMPLETE')]
        if self.parent is None:
            self.initialize(stack_template=stack_template)
        self.heat.stacks.create.return_value = {'stack': get_stack().to_dict()}
        self.heat.stacks.get = mock.MagicMock(side_effect=defaults)
        rsrc = self.parent['remote_stack']
        scheduler.TaskRunner(rsrc.create)()
        return rsrc

    def test_create_remote_stack_default_region(self):
        parent, rsrc = self.create_parent_stack()
        self.assertEqual((rsrc.INIT, rsrc.COMPLETE), rsrc.state)
        self.assertEqual(self.this_region, rsrc._region_name)
        ctx = rsrc.properties.get('context')
        self.assertIsNone(ctx)
        self.assertIsNone(rsrc.validate())

    def test_create_remote_stack_this_region(self):
        parent, rsrc = self.create_parent_stack(remote_region=self.this_region)
        self.assertEqual((rsrc.INIT, rsrc.COMPLETE), rsrc.state)
        self.assertEqual(self.this_region, rsrc._region_name)
        ctx = rsrc.properties.get('context')
        self.assertEqual(self.this_region, ctx['region_name'])
        self.assertIsNone(rsrc.validate())

    def test_create_remote_stack_that_region(self):
        parent, rsrc = self.create_parent_stack(remote_region=self.that_region)
        self.assertEqual((rsrc.INIT, rsrc.COMPLETE), rsrc.state)
        self.assertEqual(self.that_region, rsrc._region_name)
        ctx = rsrc.properties.get('context')
        self.assertEqual(self.that_region, ctx['region_name'])
        self.assertIsNone(rsrc.validate())

    def test_create_remote_stack_bad_region(self):
        parent, rsrc = self.create_parent_stack(remote_region=self.bad_region)
        self.assertEqual((rsrc.INIT, rsrc.COMPLETE), rsrc.state)
        self.assertEqual(self.bad_region, rsrc._region_name)
        ctx = rsrc.properties.get('context')
        self.assertEqual(self.bad_region, ctx['region_name'])
        ex = self.assertRaises(exception.StackValidationFailed, rsrc.validate)
        msg = 'Cannot establish connection to Heat endpoint at region "%s"' % self.bad_region
        self.assertIn(msg, str(ex))

    def test_remote_validation_failed(self):
        parent, rsrc = self.create_parent_stack(remote_region=self.that_region, custom_template=bad_template)
        self.assertEqual((rsrc.INIT, rsrc.COMPLETE), rsrc.state)
        self.assertEqual(self.that_region, rsrc._region_name)
        ctx = rsrc.properties.get('context')
        self.assertEqual(self.that_region, ctx['region_name'])
        heat = rsrc._context().clients.client('heat')
        heat.stacks.validate = mock.MagicMock(side_effect=exc.HTTPBadRequest)
        ex = self.assertRaises(exception.StackValidationFailed, rsrc.validate)
        msg = 'Failed validating stack template using Heat endpoint at region "%s"' % self.that_region
        self.assertIn(msg, str(ex))

    def test_create(self):
        rsrc = self.create_remote_stack()
        self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
        self.assertEqual('c8a19429-7fde-47ea-a42f-40045488226c', rsrc.resource_id)
        env = environment.get_child_environment(rsrc.stack.env, {'name': 'foo'})
        args = {'stack_name': rsrc.physical_resource_name(), 'template': template_format.parse(remote_template), 'timeout_mins': 60, 'disable_rollback': True, 'parameters': {'name': 'foo'}, 'files': self.files, 'environment': env.user_env_as_dict()}
        self.heat.stacks.create.assert_called_with(**args)
        self.assertEqual(2, len(self.heat.stacks.get.call_args_list))

    def _create_with_remote_credential(self, credential_secret_id=None, ca_cert=None, insecure=False):
        t = template_format.parse(parent_stack_template)
        properties = t['resources']['remote_stack']['properties']
        if credential_secret_id:
            properties['context']['credential_secret_id'] = credential_secret_id
        if ca_cert:
            properties['context']['ca_cert'] = ca_cert
        if insecure:
            properties['context']['insecure'] = insecure
        t = json.dumps(t)
        self.patchobject(policy.Enforcer, 'check_is_admin')
        rsrc = self.create_remote_stack(stack_template=t)
        env = environment.get_child_environment(rsrc.stack.env, {'name': 'foo'})
        args = {'stack_name': rsrc.physical_resource_name(), 'template': template_format.parse(remote_template), 'timeout_mins': 60, 'disable_rollback': True, 'parameters': {'name': 'foo'}, 'files': self.files, 'environment': env.user_env_as_dict()}
        self.heat.stacks.create.assert_called_with(**args)
        self.assertEqual(2, len(self.heat.stacks.get.call_args_list))
        rsrc.validate()
        return rsrc

    @mock.patch('heat.engine.clients.os.barbican.BarbicanClientPlugin.get_secret_payload_by_ref')
    def test_create_with_credential_secret_id(self, m_gsbr):
        secret = '{"auth_type": "v3applicationcredential", "auth": {"auth_url": "http://192.168.1.101/identity/v3", "application_credential_id": "9dfa187e5a354484bf9c49a2b674333a", "application_credential_secret": "sec"} }'
        m_gsbr.return_value = secret
        self.m_plugin = mock.Mock()
        self.m_loader = self.patchobject(ks_loading, 'get_plugin_loader', return_value=self.m_plugin)
        self._create_with_remote_credential('cred_2')
        self.assertEqual([mock.call(secret_ref='secrets/cred_2')] * 2, m_gsbr.call_args_list)
        expected_load_options = [mock.call(application_credential_id='9dfa187e5a354484bf9c49a2b674333a', application_credential_secret='sec', auth_url='http://192.168.1.101/identity/v3')] * 2
        self.assertEqual(expected_load_options, self.m_plugin.load_from_options.call_args_list)

    def test_create_with_ca_cert(self):
        ca_cert = '-----BEGIN CERTIFICATE----- A CA CERT -----END CERTIFICATE-----'
        rsrc = self._create_with_remote_credential(ca_cert=ca_cert)
        self.assertEqual(ca_cert, rsrc._cacert)
        self.assertEqual(ca_cert, rsrc.cacert)
        self.assertIn('/tmp/', rsrc._ssl_verify)

    def test_create_with_insecure(self):
        rsrc = self._create_with_remote_credential(insecure=True)
        self.assertFalse(rsrc._ssl_verify)

    def test_create_failed(self):
        returns = [get_stack(stack_status='CREATE_IN_PROGRESS'), get_stack(stack_status='CREATE_FAILED', stack_status_reason='Remote stack creation failed')]
        if self.parent is None:
            self.initialize()
        self.heat.stacks.create.return_value = {'stack': get_stack().to_dict()}
        self.heat.stacks.get = mock.MagicMock(side_effect=returns)
        rsrc = self.parent['remote_stack']
        error = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(rsrc.create))
        error_msg = 'ResourceInError: resources.remote_stack: Went to status CREATE_FAILED due to "Remote stack creation failed"'
        self.assertEqual(error_msg, str(error))
        self.assertEqual((rsrc.CREATE, rsrc.FAILED), rsrc.state)

    def test_delete(self):
        returns = [get_stack(stack_status='DELETE_IN_PROGRESS'), get_stack(stack_status='DELETE_COMPLETE')]
        rsrc = self.create_remote_stack()
        self.heat.stacks.get = mock.MagicMock(side_effect=returns)
        self.heat.stacks.delete = mock.MagicMock()
        remote_stack_id = rsrc.resource_id
        scheduler.TaskRunner(rsrc.delete)()
        self.assertEqual((rsrc.DELETE, rsrc.COMPLETE), rsrc.state)
        self.heat.stacks.delete.assert_called_with(stack_id=remote_stack_id)

    def test_delete_already_gone(self):
        rsrc = self.create_remote_stack()
        self.heat.stacks.delete = mock.MagicMock(side_effect=exc.HTTPNotFound())
        self.heat.stacks.get = mock.MagicMock(side_effect=exc.HTTPNotFound())
        remote_stack_id = rsrc.resource_id
        scheduler.TaskRunner(rsrc.delete)()
        self.assertEqual((rsrc.DELETE, rsrc.COMPLETE), rsrc.state)
        self.heat.stacks.delete.assert_called_with(stack_id=remote_stack_id)

    def test_delete_failed(self):
        returns = [get_stack(stack_status='DELETE_IN_PROGRESS'), get_stack(stack_status='DELETE_FAILED', stack_status_reason='Remote stack deletion failed')]
        rsrc = self.create_remote_stack()
        self.heat.stacks.get = mock.MagicMock(side_effect=returns)
        self.heat.stacks.delete = mock.MagicMock()
        remote_stack_id = rsrc.resource_id
        error = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(rsrc.delete))
        error_msg = 'ResourceInError: resources.remote_stack: Went to status DELETE_FAILED due to "Remote stack deletion failed"'
        self.assertIn(error_msg, str(error))
        self.assertEqual((rsrc.DELETE, rsrc.FAILED), rsrc.state)
        self.heat.stacks.delete.assert_called_with(stack_id=remote_stack_id)
        self.assertEqual(rsrc.resource_id, remote_stack_id)

    def test_attribute(self):
        rsrc = self.create_remote_stack()
        outputs = [{'output_key': 'foo', 'output_value': 'bar'}]
        created_stack = get_stack(stack_name='stack1', outputs=outputs)
        self.heat.stacks.get = mock.MagicMock(return_value=created_stack)
        self.assertEqual('stack1', rsrc.FnGetAtt('stack_name'))
        self.assertEqual('bar', rsrc.FnGetAtt('outputs')['foo'])
        self.heat.stacks.get.assert_called_with(stack_id='c8a19429-7fde-47ea-a42f-40045488226c')

    def test_attribute_failed(self):
        rsrc = self.create_remote_stack()
        error = self.assertRaises(exception.InvalidTemplateAttribute, rsrc.FnGetAtt, 'non-existent_property')
        self.assertEqual('The Referenced Attribute (remote_stack non-existent_property) is incorrect.', str(error))

    def test_snapshot(self):
        stacks = [get_stack(stack_status='SNAPSHOT_IN_PROGRESS'), get_stack(stack_status='SNAPSHOT_COMPLETE')]
        snapshot = {'id': 'a29bc9e25aa44f99a9a3d59cd5b0e263', 'status': 'IN_PROGRESS'}
        rsrc = self.create_remote_stack()
        self.heat.stacks.get = mock.MagicMock(side_effect=stacks)
        self.heat.stacks.snapshot = mock.MagicMock(return_value=snapshot)
        scheduler.TaskRunner(rsrc.snapshot)()
        self.assertEqual((rsrc.SNAPSHOT, rsrc.COMPLETE), rsrc.state)
        self.assertEqual('a29bc9e25aa44f99a9a3d59cd5b0e263', rsrc.data().get('snapshot_id'))
        self.heat.stacks.snapshot.assert_called_with(stack_id=rsrc.resource_id)

    def test_restore(self):
        snapshot = {'id': 'a29bc9e25aa44f99a9a3d59cd5b0e263', 'status': 'IN_PROGRESS'}
        remote_stack = mock.MagicMock()
        remote_stack.action = 'SNAPSHOT'
        remote_stack.status = 'COMPLETE'
        parent, rsrc = self.create_parent_stack()
        rsrc.action = rsrc.SNAPSHOT
        heat = rsrc._context().clients.client('heat')
        heat.stacks.snapshot = mock.MagicMock(return_value=snapshot)
        heat.stacks.get = mock.MagicMock(return_value=remote_stack)
        scheduler.TaskRunner(parent.snapshot, None)()
        self.assertEqual((parent.SNAPSHOT, parent.COMPLETE), parent.state)
        data = parent.prepare_abandon()
        remote_stack_snapshot = {'snapshot': {'id': 'a29bc9e25aa44f99a9a3d59cd5b0e263', 'status': 'COMPLETE', 'data': {'files': data['files'], 'environment': data['environment'], 'template': template_format.parse(data['files']['remote_template.yaml'])}}}
        fake_snapshot = collections.namedtuple('Snapshot', ('data', 'stack_id'))(data, parent.id)
        heat.stacks.snapshot_show = mock.MagicMock(return_value=remote_stack_snapshot)
        self.patchobject(rsrc, 'update').return_value = None
        rsrc.action = rsrc.UPDATE
        rsrc.status = rsrc.COMPLETE
        remote_stack.action = 'UPDATE'
        parent.restore(fake_snapshot)
        self.assertEqual((parent.RESTORE, parent.COMPLETE), parent.state)

    def test_check(self):
        stacks = [get_stack(stack_status='CHECK_IN_PROGRESS'), get_stack(stack_status='CHECK_COMPLETE')]
        rsrc = self.create_remote_stack()
        self.heat.stacks.get = mock.MagicMock(side_effect=stacks)
        self.heat.actions.check = mock.MagicMock()
        scheduler.TaskRunner(rsrc.check)()
        self.assertEqual((rsrc.CHECK, rsrc.COMPLETE), rsrc.state)
        self.heat.actions.check.assert_called_with(stack_id=rsrc.resource_id)

    def test_check_failed(self):
        returns = [get_stack(stack_status='CHECK_IN_PROGRESS'), get_stack(stack_status='CHECK_FAILED', stack_status_reason='Remote stack check failed')]
        rsrc = self.create_remote_stack()
        self.heat.stacks.get = mock.MagicMock(side_effect=returns)
        self.heat.actions.resume = mock.MagicMock()
        error = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(rsrc.check))
        error_msg = 'ResourceInError: resources.remote_stack: Went to status CHECK_FAILED due to "Remote stack check failed"'
        self.assertEqual(error_msg, str(error))
        self.assertEqual((rsrc.CHECK, rsrc.FAILED), rsrc.state)
        self.heat.actions.check.assert_called_with(stack_id=rsrc.resource_id)

    def test_resume(self):
        stacks = [get_stack(stack_status='RESUME_IN_PROGRESS'), get_stack(stack_status='RESUME_COMPLETE')]
        rsrc = self.create_remote_stack()
        rsrc.action = rsrc.SUSPEND
        self.heat.stacks.get = mock.MagicMock(side_effect=stacks)
        self.heat.actions.resume = mock.MagicMock()
        scheduler.TaskRunner(rsrc.resume)()
        self.assertEqual((rsrc.RESUME, rsrc.COMPLETE), rsrc.state)
        self.heat.actions.resume.assert_called_with(stack_id=rsrc.resource_id)

    def test_resume_failed(self):
        returns = [get_stack(stack_status='RESUME_IN_PROGRESS'), get_stack(stack_status='RESUME_FAILED', stack_status_reason='Remote stack resume failed')]
        rsrc = self.create_remote_stack()
        rsrc.action = rsrc.SUSPEND
        self.heat.stacks.get = mock.MagicMock(side_effect=returns)
        self.heat.actions.resume = mock.MagicMock()
        error = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(rsrc.resume))
        error_msg = 'ResourceInError: resources.remote_stack: Went to status RESUME_FAILED due to "Remote stack resume failed"'
        self.assertEqual(error_msg, str(error))
        self.assertEqual((rsrc.RESUME, rsrc.FAILED), rsrc.state)
        self.heat.actions.resume.assert_called_with(stack_id=rsrc.resource_id)

    def test_resume_failed_not_created(self):
        self.initialize()
        rsrc = self.parent['remote_stack']
        rsrc.action = rsrc.SUSPEND
        error = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(rsrc.resume))
        error_msg = 'Error: resources.remote_stack: Cannot resume remote_stack, resource not found'
        self.assertEqual(error_msg, str(error))
        self.assertEqual((rsrc.RESUME, rsrc.FAILED), rsrc.state)

    def test_suspend(self):
        stacks = [get_stack(stack_status='SUSPEND_IN_PROGRESS'), get_stack(stack_status='SUSPEND_COMPLETE')]
        rsrc = self.create_remote_stack()
        self.heat.stacks.get = mock.MagicMock(side_effect=stacks)
        self.heat.actions.suspend = mock.MagicMock()
        scheduler.TaskRunner(rsrc.suspend)()
        self.assertEqual((rsrc.SUSPEND, rsrc.COMPLETE), rsrc.state)
        self.heat.actions.suspend.assert_called_with(stack_id=rsrc.resource_id)

    def test_suspend_failed(self):
        stacks = [get_stack(stack_status='SUSPEND_IN_PROGRESS'), get_stack(stack_status='SUSPEND_FAILED', stack_status_reason='Remote stack suspend failed')]
        rsrc = self.create_remote_stack()
        self.heat.stacks.get = mock.MagicMock(side_effect=stacks)
        self.heat.actions.suspend = mock.MagicMock()
        error = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(rsrc.suspend))
        error_msg = 'ResourceInError: resources.remote_stack: Went to status SUSPEND_FAILED due to "Remote stack suspend failed"'
        self.assertEqual(error_msg, str(error))
        self.assertEqual((rsrc.SUSPEND, rsrc.FAILED), rsrc.state)
        self.heat.actions.suspend.assert_has_calls([])

    def test_suspend_failed_not_created(self):
        self.initialize()
        rsrc = self.parent['remote_stack']
        self.heat.actions.suspend = mock.MagicMock()
        error = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(rsrc.suspend))
        error_msg = 'Error: resources.remote_stack: Cannot suspend remote_stack, resource not found'
        self.assertEqual(error_msg, str(error))
        self.assertEqual((rsrc.SUSPEND, rsrc.FAILED), rsrc.state)
        self.heat.actions.suspend.assert_has_calls([])

    def test_update(self):
        stacks = [get_stack(stack_status='UPDATE_IN_PROGRESS'), get_stack(stack_status='UPDATE_COMPLETE')]
        rsrc = self.create_remote_stack()
        props = dict(rsrc.properties)
        props['parameters']['name'] = 'bar'
        update_snippet = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), props)
        self.heat.stacks.get = mock.MagicMock(side_effect=stacks)
        scheduler.TaskRunner(rsrc.update, update_snippet)()
        self.assertEqual((rsrc.UPDATE, rsrc.COMPLETE), rsrc.state)
        self.assertEqual('bar', rsrc.properties.get('parameters')['name'])
        env = environment.get_child_environment(rsrc.stack.env, {'name': 'bar'})
        fields = {'stack_id': rsrc.resource_id, 'template': template_format.parse(remote_template), 'timeout_mins': 60, 'disable_rollback': True, 'parameters': {'name': 'bar'}, 'files': self.files, 'environment': env.user_env_as_dict()}
        self.heat.stacks.update.assert_called_with(**fields)
        self.assertEqual(2, len(self.heat.stacks.get.call_args_list))

    def test_update_with_replace(self):
        rsrc = self.create_remote_stack()
        props = dict(rsrc.properties)
        props['context']['region_name'] = 'RegionOne'
        update_snippet = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), props)
        self.assertRaises(resource.UpdateReplace, scheduler.TaskRunner(rsrc.update, update_snippet))

    def test_update_failed(self):
        stacks = [get_stack(stack_status='UPDATE_IN_PROGRESS'), get_stack(stack_status='UPDATE_FAILED', stack_status_reason='Remote stack update failed')]
        rsrc = self.create_remote_stack()
        props = dict(rsrc.properties)
        props['parameters']['name'] = 'bar'
        update_snippet = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), props)
        self.heat.stacks.get = mock.MagicMock(side_effect=stacks)
        error = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(rsrc.update, update_snippet))
        error_msg = _('ResourceInError: resources.remote_stack: Went to status UPDATE_FAILED due to "Remote stack update failed"')
        self.assertEqual(error_msg, str(error))
        self.assertEqual((rsrc.UPDATE, rsrc.FAILED), rsrc.state)
        self.assertEqual(2, len(self.heat.stacks.get.call_args_list))

    def test_update_no_change(self):
        stacks = [get_stack(stack_status='UPDATE_IN_PROGRESS'), get_stack(stack_status='UPDATE_COMPLETE')]
        rsrc = self.create_remote_stack()
        props = dict(rsrc.properties)
        update_snippet = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), props)
        self.heat.stacks.get = mock.MagicMock(side_effect=stacks)
        scheduler.TaskRunner(rsrc.update, update_snippet)()
        self.assertEqual((rsrc.UPDATE, rsrc.COMPLETE), rsrc.state)

    def test_remote_stack_refid(self):
        t = template_format.parse(parent_stack_template)
        stack = utils.parse_stack(t)
        rsrc = stack['remote_stack']
        rsrc.resource_id = 'xyz'
        self.assertEqual('xyz', rsrc.FnGetRefId())

    def test_remote_stack_refid_convergence_cache_data(self):
        t = template_format.parse(parent_stack_template)
        cache_data = {'remote_stack': node_data.NodeData.from_dict({'uuid': mock.ANY, 'id': mock.ANY, 'action': 'CREATE', 'status': 'COMPLETE', 'reference_id': 'convg_xyz'})}
        stack = utils.parse_stack(t, cache_data=cache_data)
        rsrc = stack.defn['remote_stack']
        self.assertEqual('convg_xyz', rsrc.FnGetRefId())

    def test_update_in_check_failed_state(self):
        rsrc = self.create_remote_stack()
        rsrc.state_set(rsrc.CHECK, rsrc.FAILED)
        props = dict(rsrc.properties)
        update_snippet = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), props)
        self.assertRaises(resource.UpdateReplace, scheduler.TaskRunner(rsrc.update, update_snippet))