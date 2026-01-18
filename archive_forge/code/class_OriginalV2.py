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
class OriginalV2(BaseV2):

    def new_client(self):
        with self.deprecations.expect_deprecations_here():
            return v2_client.Client(username=uuid.uuid4().hex, user_id=self.user_id, token=uuid.uuid4().hex, tenant_name=uuid.uuid4().hex, auth_url=self.TEST_URL, endpoint=self.TEST_URL)