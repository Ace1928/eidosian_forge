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
class _ProjectUserTagTests(object):

    def test_user_can_get_tag_for_project(self):
        tag = uuid.uuid4().hex
        PROVIDERS.resource_api.create_project_tag(self.project_id, tag)
        with self.test_client() as c:
            c.get('/v3/projects/%s/tags/%s' % (self.project_id, tag), headers=self.headers, expected_status_code=http.client.NO_CONTENT)

    def test_user_can_list_tags_for_project(self):
        tag = uuid.uuid4().hex
        PROVIDERS.resource_api.create_project_tag(self.project_id, tag)
        with self.test_client() as c:
            r = c.get('/v3/projects/%s/tags' % self.project_id, headers=self.headers)
            self.assertTrue(len(r.json['tags']) == 1)
            self.assertEqual(tag, r.json['tags'][0])

    def test_user_cannot_create_tag_for_other_project(self):
        project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=CONF.identity.default_domain_id))
        tag = uuid.uuid4().hex
        with self.test_client() as c:
            c.put('/v3/projects/%s/tags/%s' % (project['id'], tag), headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_update_tag_for_other_project(self):
        project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=CONF.identity.default_domain_id))
        tag = uuid.uuid4().hex
        PROVIDERS.resource_api.create_project_tag(project['id'], tag)
        update = {'tags': [uuid.uuid4().hex]}
        with self.test_client() as c:
            c.put('/v3/projects/%s/tags' % project['id'], headers=self.headers, json=update, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_delete_tag_for_other_project(self):
        project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=CONF.identity.default_domain_id))
        tag = uuid.uuid4().hex
        PROVIDERS.resource_api.create_project_tag(project['id'], tag)
        with self.test_client() as c:
            c.delete('/v3/projects/%s/tags/%s' % (project['id'], tag), headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_get_tag_for_other_project(self):
        project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=CONF.identity.default_domain_id))
        tag = uuid.uuid4().hex
        PROVIDERS.resource_api.create_project_tag(project['id'], tag)
        with self.test_client() as c:
            c.get('/v3/projects/%s/tags/%s' % (project['id'], tag), headers=self.headers, expected_status_code=http.client.FORBIDDEN)

    def test_user_cannot_list_tags_for_other_project(self):
        project = PROVIDERS.resource_api.create_project(uuid.uuid4().hex, unit.new_project_ref(domain_id=CONF.identity.default_domain_id))
        tag = uuid.uuid4().hex
        PROVIDERS.resource_api.create_project_tag(project['id'], tag)
        with self.test_client() as c:
            c.get('/v3/projects/%s/tags' % project['id'], headers=self.headers, expected_status_code=http.client.FORBIDDEN)