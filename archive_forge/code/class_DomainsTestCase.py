import uuid
from keystoneauth1.exceptions import http
from keystoneclient.tests.functional import base
from keystoneclient.tests.functional.v3 import client_fixtures as fixtures
class DomainsTestCase(base.V3ClientTestCase):

    def check_domain(self, domain, domain_ref=None):
        self.assertIsNotNone(domain.id)
        self.assertIn('self', domain.links)
        self.assertIn('/domains/' + domain.id, domain.links['self'])
        if domain_ref:
            self.assertEqual(domain_ref['name'], domain.name)
            self.assertEqual(domain_ref['enabled'], domain.enabled)
            if hasattr(domain_ref, 'description'):
                self.assertEqual(domain_ref['description'], domain.description)
        else:
            self.assertIsNotNone(domain.name)
            self.assertIsNotNone(domain.enabled)

    def test_create_domain(self):
        domain_ref = {'name': fixtures.RESOURCE_NAME_PREFIX + uuid.uuid4().hex, 'description': uuid.uuid4().hex, 'enabled': True}
        domain = self.client.domains.create(**domain_ref)
        self.check_domain(domain, domain_ref)
        self.addCleanup(self.client.domains.delete, domain)
        self.addCleanup(self.client.domains.update, domain, enabled=False)

    def test_get_domain(self):
        domain_id = self.project_domain_id
        domain_ret = self.client.domains.get(domain_id)
        self.check_domain(domain_ret)

    def test_list_domains(self):
        domain_one = fixtures.Domain(self.client)
        self.useFixture(domain_one)
        domain_two = fixtures.Domain(self.client)
        self.useFixture(domain_two)
        domains = self.client.domains.list()
        for domain in domains:
            self.check_domain(domain)
        self.assertIn(domain_one.entity, domains)
        self.assertIn(domain_two.entity, domains)

    def test_update_domain(self):
        domain = fixtures.Domain(self.client)
        self.useFixture(domain)
        new_description = uuid.uuid4().hex
        domain_ret = self.client.domains.update(domain.id, description=new_description)
        domain.ref.update({'description': new_description})
        self.check_domain(domain_ret, domain.ref)

    def test_delete_domain(self):
        domain = self.client.domains.create(name=uuid.uuid4().hex, description=uuid.uuid4().hex, enabled=True)
        self.assertRaises(http.Forbidden, self.client.domains.delete, domain.id)
        self.client.domains.update(domain, enabled=False)
        self.client.domains.delete(domain.id)
        self.assertRaises(http.NotFound, self.client.domains.get, domain.id)