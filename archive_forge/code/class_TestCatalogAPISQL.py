import copy
import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit.ksfixtures import database
from keystone.tests.unit import test_v3
class TestCatalogAPISQL(unit.TestCase):
    """Test for the catalog Manager against the SQL backend."""

    def setUp(self):
        super(TestCatalogAPISQL, self).setUp()
        self.useFixture(database.Database())
        self.load_backends()
        service = unit.new_service_ref()
        self.service_id = service['id']
        PROVIDERS.catalog_api.create_service(self.service_id, service)
        self.create_endpoint(service_id=self.service_id)
        PROVIDERS.resource_api.create_domain(default_fixtures.ROOT_DOMAIN['id'], default_fixtures.ROOT_DOMAIN)

    def create_endpoint(self, service_id, **kwargs):
        endpoint = unit.new_endpoint_ref(service_id=service_id, region_id=None, **kwargs)
        PROVIDERS.catalog_api.create_endpoint(endpoint['id'], endpoint)
        return endpoint

    def config_overrides(self):
        super(TestCatalogAPISQL, self).config_overrides()
        self.config_fixture.config(group='catalog', driver='sql')

    def test_get_catalog_ignores_endpoints_with_invalid_urls(self):
        user_id = uuid.uuid4().hex
        domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain['id'], domain)
        project = unit.new_project_ref(domain_id=domain['id'])
        PROVIDERS.resource_api.create_project(project['id'], project)
        catalog = PROVIDERS.catalog_api.get_v3_catalog(user_id, project['id'])
        self.assertEqual(1, len(catalog[0]['endpoints']))
        self.assertEqual(1, len(PROVIDERS.catalog_api.list_endpoints()))
        self.create_endpoint(self.service_id, url='http://keystone/%(project_id)')
        self.create_endpoint(self.service_id, url='http://keystone/%(you_wont_find_me)s')
        catalog = PROVIDERS.catalog_api.get_v3_catalog(user_id, project['id'])
        self.assertEqual(1, len(catalog[0]['endpoints']))
        self.assertEqual(3, len(PROVIDERS.catalog_api.list_endpoints()))
        self.create_endpoint(self.service_id, url='http://keystone/%(project_id)s')
        catalog = PROVIDERS.catalog_api.get_v3_catalog(user_id, project['id'])
        self.assertThat(catalog[0]['endpoints'], matchers.HasLength(2))
        project_id = None
        catalog = PROVIDERS.catalog_api.get_v3_catalog(user_id, project_id)
        self.assertThat(catalog[0]['endpoints'], matchers.HasLength(1))

    def test_get_catalog_always_returns_service_name(self):
        user_id = uuid.uuid4().hex
        domain = unit.new_domain_ref()
        PROVIDERS.resource_api.create_domain(domain['id'], domain)
        project = unit.new_project_ref(domain_id=domain['id'])
        PROVIDERS.resource_api.create_project(project['id'], project)
        named_svc = unit.new_service_ref()
        PROVIDERS.catalog_api.create_service(named_svc['id'], named_svc)
        self.create_endpoint(service_id=named_svc['id'])
        unnamed_svc = unit.new_service_ref(name=None)
        del unnamed_svc['name']
        PROVIDERS.catalog_api.create_service(unnamed_svc['id'], unnamed_svc)
        self.create_endpoint(service_id=unnamed_svc['id'])
        catalog = PROVIDERS.catalog_api.get_v3_catalog(user_id, project['id'])
        named_endpoint = [ep for ep in catalog if ep['type'] == named_svc['type']][0]
        self.assertEqual(named_svc['name'], named_endpoint['name'])
        unnamed_endpoint = [ep for ep in catalog if ep['type'] == unnamed_svc['type']][0]
        self.assertEqual('', unnamed_endpoint['name'])