import uuid
import http.client
from oslo_serialization import jsonutils
from keystone.common.policies import project as pp
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import temporaryfile
class _SystemMemberAndReaderProjectTests(object):
    """Common default functionality for system members and system readers."""

    def test_user_cannot_create_projects(self):
        create = {'project': unit.new_project_ref(domain_id=CONF.identity.default_domain_id)}
        with self.test_client() as c:
            c.post('/v3/projects', json=create, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_update_projects(self):
        project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=CONF.identity.default_domain_id))
        update = {'project': {'description': uuid.uuid4().hex}}
        with self.test_client() as c:
            c.patch('/v3/projects/%s' % project['id'], json=update, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_update_non_existent_project_forbidden(self):
        update = {'project': {'description': uuid.uuid4().hex}}
        with self.test_client() as c:
            c.patch('/v3/projects/%s' % uuid.uuid4().hex, json=update, headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_delete_projects(self):
        project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=CONF.identity.default_domain_id))
        with self.test_client() as c:
            c.delete('/v3/projects/%s' % project['id'], headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_delete_non_existent_project_forbidden(self):
        with self.test_client() as c:
            c.delete('/v3/projects/%s' % uuid.uuid4().hex, headers=self.headers, expected_status_code=http.client.FORBIDDEN)