from unittest import mock
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack.orchestration.v1 import _proxy
from openstack.orchestration.v1 import resource
from openstack.orchestration.v1 import software_config as sc
from openstack.orchestration.v1 import software_deployment as sd
from openstack.orchestration.v1 import stack
from openstack.orchestration.v1 import stack_environment
from openstack.orchestration.v1 import stack_event
from openstack.orchestration.v1 import stack_files
from openstack.orchestration.v1 import stack_template
from openstack.orchestration.v1 import template
from openstack import proxy
from openstack.tests.unit import test_proxy_base
class TestOrchestrationTemplate(TestOrchestrationProxy):

    @mock.patch.object(template.Template, 'validate')
    def test_validate_template(self, mock_validate):
        tmpl = mock.Mock()
        env = mock.Mock()
        tmpl_url = 'A_URI'
        ignore_errors = 'a_string'
        res = self.proxy.validate_template(tmpl, env, tmpl_url, ignore_errors)
        mock_validate.assert_called_once_with(self.proxy, tmpl, environment=env, template_url=tmpl_url, ignore_errors=ignore_errors)
        self.assertEqual(mock_validate.return_value, res)

    def test_validate_template_no_env(self):
        tmpl = 'openstack/tests/unit/orchestration/v1/hello_world.yaml'
        res = self.proxy.read_env_and_templates(tmpl)
        self.assertIsInstance(res, dict)
        self.assertIsInstance(res['files'], dict)

    def test_validate_template_invalid_request(self):
        err = self.assertRaises(exceptions.InvalidRequest, self.proxy.validate_template, None, template_url=None)
        self.assertEqual("'template_url' must be specified when template is None", str(err))