import collections
from unittest import mock
from heatclient import exc
from heatclient.osc.v1 import stack_failures
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
class ListStackFailuresTest(orchestration_fakes.TestOrchestrationv1):

    def setUp(self):
        super(ListStackFailuresTest, self).setUp()
        self.cmd = stack_failures.ListStackFailures(self.app, None)
        self.cmd.heat_client = self.app.client_manager.orchestration
        self.stack_client = self.app.client_manager.orchestration.stacks
        self.resource_client = self.app.client_manager.orchestration.resources
        self.software_deployments_client = self.app.client_manager.orchestration.software_deployments
        self.stack = mock.MagicMock(id='123', status='FAILED', stack_name='stack')
        self.stack_client.get.return_value = self.stack
        self.failed_template_resource = mock.MagicMock(physical_resource_id='aaaa', resource_type='My::TemplateResource', resource_status='CREATE_FAILED', links=[{'rel': 'nested'}], resource_name='my_templateresource', resource_status_reason='All gone Pete Tong', logical_resource_id='my_templateresource')
        self.failed_resource = mock.MagicMock(physical_resource_id='cccc', resource_type='OS::Nova::Server', resource_status='CREATE_FAILED', links=[], resource_name='my_server', resource_status_reason='All gone Pete Tong', logical_resource_id='my_server')
        self.other_failed_template_resource = mock.MagicMock(physical_resource_id='dddd', resource_type='My::OtherTemplateResource', resource_status='CREATE_FAILED', links=[{'rel': 'nested'}], resource_name='my_othertemplateresource', resource_status_reason='RPC timeout', logical_resource_id='my_othertemplateresource')
        self.working_resource = mock.MagicMock(physical_resource_id='bbbb', resource_type='OS::Nova::Server', resource_status='CREATE_COMPLETE', resource_name='my_server')
        self.failed_deployment_resource = mock.MagicMock(physical_resource_id='eeee', resource_type='OS::Heat::SoftwareDeployment', resource_status='CREATE_FAILED', links=[], resource_name='my_deployment', resource_status_reason='Returned deploy_statuscode 1', logical_resource_id='my_deployment')
        self.failed_deployment = mock.MagicMock(id='eeee', output_values={'deploy_statuscode': '1', 'deploy_stderr': 'It broke', 'deploy_stdout': '1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n11\n12'})
        self.software_deployments_client.get.return_value = self.failed_deployment

    def test_build_failed_none(self):
        self.stack = mock.MagicMock(id='123', status='COMPLETE', stack_name='stack')
        failures = self.cmd._build_failed_resources('stack')
        expected = collections.OrderedDict()
        self.assertEqual(expected, failures)

    def test_build_failed_resources(self):
        self.resource_client.list.side_effect = [[self.failed_template_resource, self.other_failed_template_resource, self.working_resource], [self.failed_resource], []]
        failures = self.cmd._build_failed_resources('stack')
        expected = collections.OrderedDict()
        expected['stack.my_templateresource.my_server'] = self.failed_resource
        expected['stack.my_othertemplateresource'] = self.other_failed_template_resource
        self.assertEqual(expected, failures)

    def test_build_failed_resources_not_found(self):
        self.resource_client.list.side_effect = [[self.failed_template_resource, self.other_failed_template_resource, self.working_resource], exc.HTTPNotFound(), []]
        failures = self.cmd._build_failed_resources('stack')
        expected = collections.OrderedDict()
        expected['stack.my_templateresource'] = self.failed_template_resource
        expected['stack.my_othertemplateresource'] = self.other_failed_template_resource
        self.assertEqual(expected, failures)

    def test_build_software_deployments(self):
        resources = {'stack.my_server': self.working_resource, 'stack.my_deployment': self.failed_deployment_resource}
        deployments = self.cmd._build_software_deployments(resources)
        self.assertEqual({'eeee': self.failed_deployment}, deployments)

    def test_build_software_deployments_not_found(self):
        resources = {'stack.my_server': self.working_resource, 'stack.my_deployment': self.failed_deployment_resource}
        self.software_deployments_client.get.side_effect = exc.HTTPNotFound()
        deployments = self.cmd._build_software_deployments(resources)
        self.assertEqual({}, deployments)

    def test_build_software_deployments_no_resources(self):
        resources = {}
        self.software_deployments_client.get.side_effect = exc.HTTPNotFound()
        deployments = self.cmd._build_software_deployments(resources)
        self.assertEqual({}, deployments)

    def test_list_stack_failures(self):
        self.resource_client.list.side_effect = [[self.failed_template_resource, self.other_failed_template_resource, self.working_resource, self.failed_deployment_resource], [self.failed_resource], []]
        arglist = ['stack']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.take_action(parsed_args)
        self.assertEqual(self.app.stdout.make_string(), 'stack.my_templateresource.my_server:\n  resource_type: OS::Nova::Server\n  physical_resource_id: cccc\n  status: CREATE_FAILED\n  status_reason: |\n    All gone Pete Tong\nstack.my_othertemplateresource:\n  resource_type: My::OtherTemplateResource\n  physical_resource_id: dddd\n  status: CREATE_FAILED\n  status_reason: |\n    RPC timeout\nstack.my_deployment:\n  resource_type: OS::Heat::SoftwareDeployment\n  physical_resource_id: eeee\n  status: CREATE_FAILED\n  status_reason: |\n    Returned deploy_statuscode 1\n  deploy_stdout: |\n    ...\n    3\n    4\n    5\n    6\n    7\n    8\n    9\n    10\n    11\n    12\n    (truncated, view all with --long)\n  deploy_stderr: |\n    It broke\n')

    def test_list_stack_failures_long(self):
        self.resource_client.list.side_effect = [[self.failed_template_resource, self.other_failed_template_resource, self.working_resource, self.failed_deployment_resource], [self.failed_resource], []]
        arglist = ['--long', 'stack']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.take_action(parsed_args)
        self.assertEqual(self.app.stdout.make_string(), 'stack.my_templateresource.my_server:\n  resource_type: OS::Nova::Server\n  physical_resource_id: cccc\n  status: CREATE_FAILED\n  status_reason: |\n    All gone Pete Tong\nstack.my_othertemplateresource:\n  resource_type: My::OtherTemplateResource\n  physical_resource_id: dddd\n  status: CREATE_FAILED\n  status_reason: |\n    RPC timeout\nstack.my_deployment:\n  resource_type: OS::Heat::SoftwareDeployment\n  physical_resource_id: eeee\n  status: CREATE_FAILED\n  status_reason: |\n    Returned deploy_statuscode 1\n  deploy_stdout: |\n    1\n    2\n    3\n    4\n    5\n    6\n    7\n    8\n    9\n    10\n    11\n    12\n  deploy_stderr: |\n    It broke\n')