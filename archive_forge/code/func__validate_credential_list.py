import uuid
from oslo_config import fixture as config_fixture
from keystone.common import provider_api
from keystone.credential.providers import fernet as credential_provider
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.credential.backends import sql as credential_sql
from keystone import exception
def _validate_credential_list(self, retrieved_credentials, expected_credentials):
    self.assertEqual(len(expected_credentials), len(retrieved_credentials))
    retrieved_ids = [c['id'] for c in retrieved_credentials]
    for cred in expected_credentials:
        self.assertIn(cred['id'], retrieved_ids)