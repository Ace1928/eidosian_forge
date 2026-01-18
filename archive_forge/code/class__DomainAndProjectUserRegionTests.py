import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
class _DomainAndProjectUserRegionTests(object):
    """Common default functionality for all domain and project users."""

    def test_user_cannot_create_regions(self):
        create = {'region': {'description': uuid.uuid4().hex}}
        with self.test_client() as c:
            c.post('/v3/regions', json=create, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_update_regions(self):
        region = PROVIDERS.catalog_api.create_region(unit.new_region_ref())
        with self.test_client() as c:
            update = {'region': {'description': uuid.uuid4().hex}}
            c.patch('/v3/regions/%s' % region['id'], json=update, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_delete_regions(self):
        region = PROVIDERS.catalog_api.create_region(unit.new_region_ref())
        with self.test_client() as c:
            c.delete('/v3/regions/%s' % region['id'], headers=self.headers, expected_status_code=http.client.FORBIDDEN)