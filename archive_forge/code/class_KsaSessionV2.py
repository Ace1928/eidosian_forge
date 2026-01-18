import contextlib
import os
import uuid
import warnings
import fixtures
from keystoneauth1 import fixture
from keystoneauth1 import identity as ksa_identity
from keystoneauth1 import session as ksa_session
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import testresources
from keystoneclient.auth import identity as ksc_identity
from keystoneclient.common import cms
from keystoneclient import session as ksc_session
from keystoneclient import utils
from keystoneclient.v2_0 import client as v2_client
from keystoneclient.v3 import client as v3_client
class KsaSessionV2(BaseV2):

    def new_client(self):
        t = fixture.V2Token(user_id=self.user_id)
        t.set_scope()
        s = t.add_service('identity')
        s.add_endpoint(self.TEST_URL)
        d = fixture.V2Discovery(self.TEST_URL)
        self.requests.register_uri('POST', self.TEST_URL + '/tokens', json=t)
        self.requests.register_uri('GET', self.TEST_ROOT_URL, json={'version': d})
        a = ksa_identity.V2Password(username=uuid.uuid4().hex, password=uuid.uuid4().hex, auth_url=self.TEST_URL)
        s = ksa_session.Session(auth=a)
        return v2_client.Client(session=s)