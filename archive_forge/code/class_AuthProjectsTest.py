import uuid
from keystoneauth1 import fixture
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import auth
class AuthProjectsTest(utils.ClientTestCase):

    def setUp(self):
        super(AuthProjectsTest, self).setUp()
        self.v3token = fixture.V3Token()
        self.stub_auth(json=self.v3token)
        self.stub_url('GET', [], json={'version': fixture.V3Discovery(self.TEST_URL)})

    def create_resource(self, id=None, name=None, **kwargs):
        kwargs['id'] = id or uuid.uuid4().hex
        kwargs['name'] = name or uuid.uuid4().hex
        return kwargs

    def test_get_projects(self):
        body = {'projects': [self.create_resource(), self.create_resource(), self.create_resource()]}
        self.stub_url('GET', ['auth', 'projects'], json=body)
        projects = self.client.auth.projects()
        self.assertEqual(3, len(projects))
        for p in projects:
            self.assertIsInstance(p, auth.Project)

    def test_get_domains(self):
        body = {'domains': [self.create_resource(), self.create_resource(), self.create_resource()]}
        self.stub_url('GET', ['auth', 'domains'], json=body)
        domains = self.client.auth.domains()
        self.assertEqual(3, len(domains))
        for d in domains:
            self.assertIsInstance(d, auth.Domain)

    def test_get_systems(self):
        body = {'system': [{'all': True}]}
        self.stub_url('GET', ['auth', 'system'], json=body)
        systems = self.client.auth.systems()
        system = systems[0]
        self.assertEqual(1, len(systems))
        self.assertIsInstance(system, auth.System)
        self.assertTrue(system.all)