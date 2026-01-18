import copy
from osc_lib.tests import utils as osc_lib_utils
from openstackclient import shell
from openstackclient.tests.unit.integ import base as test_base
from openstackclient.tests.unit import test_shell
class TestIntegV2ProjectID(test_base.TestInteg):

    def setUp(self):
        super(TestIntegV2ProjectID, self).setUp()
        env = {'OS_AUTH_URL': test_base.V2_AUTH_URL, 'OS_PROJECT_ID': test_shell.DEFAULT_PROJECT_ID, 'OS_USERNAME': test_shell.DEFAULT_USERNAME, 'OS_PASSWORD': test_shell.DEFAULT_PASSWORD, 'OS_IDENTITY_API_VERSION': '2'}
        self.useFixture(osc_lib_utils.EnvFixture(copy.deepcopy(env)))
        self.token = test_base.make_v2_token(self.requests_mock)

    def test_project_id_env(self):
        _shell = shell.OpenStackShell()
        _shell.run('extension list'.split())
        self.assertNotEqual(len(self.requests_mock.request_history), 0)
        self.assertEqual(test_base.V2_AUTH_URL, self.requests_mock.request_history[0].url)
        auth_req = self.requests_mock.request_history[1].json()
        self.assertEqual(test_shell.DEFAULT_PROJECT_ID, auth_req['auth']['tenantId'])

    def test_project_id_arg(self):
        _shell = shell.OpenStackShell()
        _shell.run('--os-project-id wsx extension list'.split())
        self.assertNotEqual(len(self.requests_mock.request_history), 0)
        self.assertEqual(test_base.V2_AUTH_URL, self.requests_mock.request_history[0].url)
        auth_req = self.requests_mock.request_history[1].json()
        self.assertEqual('wsx', auth_req['auth']['tenantId'])