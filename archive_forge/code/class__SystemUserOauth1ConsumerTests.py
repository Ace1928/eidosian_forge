import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone.tests.common import auth as common_auth
from keystone.tests import unit
from keystone.tests.unit import base_classes
from keystone.tests.unit import ksfixtures
class _SystemUserOauth1ConsumerTests(object):
    """Common default functionality for all system users."""

    def test_user_can_get_consumer(self):
        ref = PROVIDERS.oauth_api.create_consumer({'id': uuid.uuid4().hex})
        with self.test_client() as c:
            c.get('/v3/OS-OAUTH1/consumers/%s' % ref['id'], headers=self.headers)

    def test_user_can_list_consumers(self):
        PROVIDERS.oauth_api.create_consumer({'id': uuid.uuid4().hex})
        with self.test_client() as c:
            c.get('/v3/OS-OAUTH1/consumers', headers=self.headers)