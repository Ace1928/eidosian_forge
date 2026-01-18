import uuid
from openstack.tests import fakes
from openstack.tests.unit import base
class TestServerConsole(base.TestCase):

    def setUp(self):
        super(TestServerConsole, self).setUp()
        self.server_id = str(uuid.uuid4())
        self.server_name = self.getUniqueString('name')
        self.server = fakes.make_fake_server(server_id=self.server_id, name=self.server_name)
        self.output = self.getUniqueString('output')

    def test_get_server_console_dict(self):
        self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='POST', uri='{endpoint}/servers/{id}/action'.format(endpoint=fakes.COMPUTE_ENDPOINT, id=self.server_id), json={'output': self.output}, validate=dict(json={'os-getConsoleOutput': {'length': 5}}))])
        self.assertEqual(self.output, self.cloud.get_server_console(self.server, 5))
        self.assert_calls()

    def test_get_server_console_name_or_id(self):
        self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='GET', uri='{endpoint}/servers/detail'.format(endpoint=fakes.COMPUTE_ENDPOINT), json={'servers': [self.server]}), dict(method='POST', uri='{endpoint}/servers/{id}/action'.format(endpoint=fakes.COMPUTE_ENDPOINT, id=self.server_id), json={'output': self.output}, validate=dict(json={'os-getConsoleOutput': {}}))])
        self.assertEqual(self.output, self.cloud.get_server_console(self.server['id']))
        self.assert_calls()

    def test_get_server_console_no_console(self):
        self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='POST', uri='{endpoint}/servers/{id}/action'.format(endpoint=fakes.COMPUTE_ENDPOINT, id=self.server_id), status_code=400, validate=dict(json={'os-getConsoleOutput': {}}))])
        self.assertEqual('', self.cloud.get_server_console(self.server))
        self.assert_calls()