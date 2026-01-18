import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
class _SystemReaderAndMemberOauth1ConsumerTests(object):

    def test_user_cannot_create_consumer(self):
        with self.test_client() as c:
            c.post('/v3/OS-OAUTH1/consumers', json={'consumer': {}}, expected_status_code=http.client.FORBIDDEN, headers=self.headers)

    def test_user_cannot_update_consumer(self):
        ref = PROVIDERS.oauth_api.create_consumer({'id': uuid.uuid4().hex})
        with self.test_client() as c:
            c.patch('/v3/OS-OAUTH1/consumers/%s' % ref['id'], json={'consumer': {'description': uuid.uuid4().hex}}, expected_status_code=http.client.FORBIDDEN, headers=self.headers)

    def test_user_cannot_delete_consumer(self):
        ref = PROVIDERS.oauth_api.create_consumer({'id': uuid.uuid4().hex})
        with self.test_client() as c:
            c.delete('/v3/OS-OAUTH1/consumers/%s' % ref['id'], expected_status_code=http.client.FORBIDDEN, headers=self.headers)