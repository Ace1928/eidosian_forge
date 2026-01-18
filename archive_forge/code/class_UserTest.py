from unittest import mock
from oslo_config import cfg
from heat.common import exception
from heat.common import short_id
from heat.common import template_format
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import node_data
from heat.engine.resources.aws.iam import user
from heat.engine.resources.openstack.heat import access_policy as ap
from heat.engine import scheduler
from heat.engine import stk_defn
from heat.objects import resource_data as resource_data_object
from heat.tests import common
from heat.tests import utils
class UserTest(common.HeatTestCase):

    def setUp(self):
        super(UserTest, self).setUp()
        self.stack_name = 'test_user_stack_%s' % utils.random_name()
        self.username = '%s-CfnUser-aabbcc' % self.stack_name
        self.fc = fake_ks.FakeKeystoneClient(username=self.username)
        cfg.CONF.set_default('heat_stack_user_role', 'stack_user_role')

    def create_user(self, t, stack, resource_name, project_id, user_id='dummy_user', password=None):
        self.patchobject(user.User, 'keystone', return_value=self.fc)
        self.mock_create_project = self.patchobject(fake_ks.FakeKeystoneClient, 'create_stack_domain_project', return_value=project_id)
        resource_defns = stack.t.resource_definitions(stack)
        rsrc = user.User(resource_name, resource_defns[resource_name], stack)
        rsrc.store()
        self.patchobject(short_id, 'get_id', return_value='aabbcc')
        self.mock_create_user = self.patchobject(fake_ks.FakeKeystoneClient, 'create_stack_domain_user', return_value=user_id)
        self.assertIsNone(rsrc.validate())
        scheduler.TaskRunner(rsrc.create)()
        self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
        return rsrc

    def test_user(self):
        t = template_format.parse(user_template)
        stack = utils.parse_stack(t, stack_name=self.stack_name)
        project_id = 'stackproject'
        rsrc = self.create_user(t, stack, 'CfnUser', project_id)
        self.assertEqual('dummy_user', rsrc.resource_id)
        self.assertEqual(self.username, rsrc.FnGetRefId())
        self.assertRaises(exception.InvalidTemplateAttribute, rsrc.FnGetAtt, 'Foo')
        self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
        self.assertIsNone(rsrc.handle_suspend())
        self.assertIsNone(rsrc.handle_resume())
        rsrc.resource_id = None
        scheduler.TaskRunner(rsrc.delete)()
        self.assertEqual((rsrc.DELETE, rsrc.COMPLETE), rsrc.state)
        rsrc.resource_id = self.fc.access
        rsrc.state_set(rsrc.CREATE, rsrc.COMPLETE)
        self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
        scheduler.TaskRunner(rsrc.delete)()
        self.assertEqual((rsrc.DELETE, rsrc.COMPLETE), rsrc.state)
        rsrc.state_set(rsrc.CREATE, rsrc.COMPLETE)
        self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
        scheduler.TaskRunner(rsrc.delete)()
        self.assertEqual((rsrc.DELETE, rsrc.COMPLETE), rsrc.state)
        self.mock_create_project.assert_called_once_with(stack.id)
        self.mock_create_user.assert_called_once_with(password=None, project_id=project_id, username=self.username)

    def test_user_password(self):
        t = template_format.parse(user_template_password)
        stack = utils.parse_stack(t, stack_name=self.stack_name)
        project_id = 'stackproject'
        password = u'myP@ssW0rd'
        rsrc = self.create_user(t, stack, 'CfnUser', project_id=project_id, password=password)
        self.assertEqual('dummy_user', rsrc.resource_id)
        self.assertEqual(self.username, rsrc.FnGetRefId())
        self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
        self.mock_create_project.assert_called_once_with(stack.id)
        self.mock_create_user.assert_called_once_with(password=password, project_id=project_id, username=self.username)

    def test_user_validate_policies(self):
        t = template_format.parse(user_policy_template)
        stack = utils.parse_stack(t, stack_name=self.stack_name)
        project_id = 'stackproject'
        rsrc = self.create_user(t, stack, 'CfnUser', project_id)
        self.assertEqual('dummy_user', rsrc.resource_id)
        self.assertEqual(self.username, rsrc.FnGetRefId())
        self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
        self.assertEqual([u'WebServerAccessPolicy'], rsrc.properties['Policies'])
        self.assertTrue(rsrc._validate_policies([u'WebServerAccessPolicy']))
        self.assertFalse(rsrc._validate_policies([u'NoExistAccessPolicy']))
        self.assertFalse(rsrc._validate_policies([u'NoExistAccessPolicy', u'WikiDatabase']))
        dict_policy = {'PolicyName': 'AccessForCFNInit', 'PolicyDocument': {'Statement': [{'Effect': 'Allow', 'Action': 'cloudformation:DescribeStackResource', 'Resource': '*'}]}}
        self.assertTrue(rsrc._validate_policies([dict_policy]))
        self.mock_create_project.assert_called_once_with(stack.id)
        self.mock_create_user.assert_called_once_with(password=None, project_id=project_id, username=self.username)

    def test_user_create_bad_policies(self):
        t = template_format.parse(user_policy_template)
        t['Resources']['CfnUser']['Properties']['Policies'] = ['NoExistBad']
        stack = utils.parse_stack(t, stack_name=self.stack_name)
        resource_name = 'CfnUser'
        resource_defns = stack.t.resource_definitions(stack)
        rsrc = user.User(resource_name, resource_defns[resource_name], stack)
        self.assertRaises(exception.InvalidTemplateAttribute, rsrc.handle_create)

    def test_user_access_allowed(self):

        def mock_access_allowed(resource):
            return True if resource == 'a_resource' else False
        self.patchobject(ap.AccessPolicy, 'access_allowed', side_effect=mock_access_allowed)
        t = template_format.parse(user_policy_template)
        stack = utils.parse_stack(t, stack_name=self.stack_name)
        project_id = 'stackproject'
        rsrc = self.create_user(t, stack, 'CfnUser', project_id)
        self.assertEqual('dummy_user', rsrc.resource_id)
        self.assertEqual(self.username, rsrc.FnGetRefId())
        self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
        self.assertTrue(rsrc.access_allowed('a_resource'))
        self.assertFalse(rsrc.access_allowed('b_resource'))
        self.mock_create_project.assert_called_once_with(stack.id)
        self.mock_create_user.assert_called_once_with(password=None, project_id=project_id, username=self.username)

    def test_user_access_allowed_ignorepolicy(self):

        def mock_access_allowed(resource):
            return True if resource == 'a_resource' else False
        self.patchobject(ap.AccessPolicy, 'access_allowed', side_effect=mock_access_allowed)
        t = template_format.parse(user_policy_template)
        t['Resources']['CfnUser']['Properties']['Policies'] = ['WebServerAccessPolicy', {'an_ignored': 'policy'}]
        stack = utils.parse_stack(t, stack_name=self.stack_name)
        project_id = 'stackproject'
        rsrc = self.create_user(t, stack, 'CfnUser', project_id)
        self.assertEqual('dummy_user', rsrc.resource_id)
        self.assertEqual(self.username, rsrc.FnGetRefId())
        self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
        self.assertTrue(rsrc.access_allowed('a_resource'))
        self.assertFalse(rsrc.access_allowed('b_resource'))
        self.mock_create_project.assert_called_once_with(stack.id)
        self.mock_create_user.assert_called_once_with(password=None, project_id=project_id, username=self.username)

    def test_user_refid_rsrc_id(self):
        t = template_format.parse(user_template)
        stack = utils.parse_stack(t)
        rsrc = stack['CfnUser']
        rsrc.resource_id = 'phy-rsrc-id'
        self.assertEqual('phy-rsrc-id', rsrc.FnGetRefId())

    def test_user_refid_convg_cache_data(self):
        t = template_format.parse(user_template)
        cache_data = {'CfnUser': node_data.NodeData.from_dict({'uuid': mock.ANY, 'id': mock.ANY, 'action': 'CREATE', 'status': 'COMPLETE', 'reference_id': 'convg_xyz'})}
        stack = utils.parse_stack(t, cache_data=cache_data)
        rsrc = stack.defn['CfnUser']
        self.assertEqual('convg_xyz', rsrc.FnGetRefId())