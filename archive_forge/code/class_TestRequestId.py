import requests
import uuid
from urllib import parse as urlparse
from keystoneauth1.identity import v3
from keystoneauth1 import session
from keystoneclient.tests.unit import client_fixtures
from keystoneclient.tests.unit import utils
from keystoneclient.v3 import client
class TestRequestId(TestCase):
    resp = requests.Response()
    TEST_REQUEST_ID = uuid.uuid4().hex
    resp.headers['x-openstack-request-id'] = TEST_REQUEST_ID

    def setUp(self):
        super(TestRequestId, self).setUp()
        auth = v3.Token(auth_url='http://127.0.0.1:5000', token=self.TEST_TOKEN)
        session_ = session.Session(auth=auth)
        self.client = client.Client(session=session_, include_metadata='True')._adapter