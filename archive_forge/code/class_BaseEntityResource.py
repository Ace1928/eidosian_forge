from unittest import mock
from keystoneauth1 import session
from requests_mock.contrib import fixture
import testtools
from barbicanclient import client
from barbicanclient import exceptions
from barbicanclient.exceptions import UnsupportedVersion
from barbicanclient.tests.utils import get_server_supported_versions
from barbicanclient.tests.utils import get_version_endpoint
from barbicanclient.tests.utils import mock_session
from barbicanclient.tests.utils import mock_session_get
from barbicanclient.tests.utils import mock_session_get_endpoint
class BaseEntityResource(TestClient):

    def _setUp(self, entity, entity_id='abcd1234-eabc-5678-9abc-abcdef012345'):
        super(BaseEntityResource, self).setUp()
        self.entity = entity
        self.entity_id = entity_id
        self.entity_base = self.endpoint + '/v1/' + self.entity
        self.entity_href = self.entity_base + '/' + self.entity_id
        self.entity_payload_href = self.entity_href + '/payload'
        self.client = client.Client(endpoint=self.endpoint, project_id=self.project_id)