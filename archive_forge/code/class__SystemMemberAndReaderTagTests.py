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
class _SystemMemberAndReaderTagTests(object):

    def test_user_cannot_create_project_tag(self):
        project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=CONF.identity.default_domain_id))
        tag = uuid.uuid4().hex
        with self.test_client() as c:
            c.put('/v3/projects/%s/tags/%s' % (project['id'], tag), headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_update_project_tag(self):
        project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=CONF.identity.default_domain_id))
        tag = uuid.uuid4().hex
        PROVIDERS.resource_api.create_project_tag(project['id'], tag)
        update = {'tags': [uuid.uuid4().hex]}
        with self.test_client() as c:
            c.put('/v3/projects/%s/tags' % project['id'], headers=self.headers, json=update, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_delete_project_tag(self):
        project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=CONF.identity.default_domain_id))
        tag = uuid.uuid4().hex
        PROVIDERS.resource_api.create_project_tag(project['id'], tag)
        with self.test_client() as c:
            c.delete('/v3/projects/%s/tags/%s' % (project['id'], tag), headers=self.headers, expected_status_code=http.client.FORBIDDEN)