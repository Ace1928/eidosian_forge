import copy
import uuid
from oslo_serialization import jsonutils
from keystoneauth1 import session as auth_session
from keystoneclient.auth import token_endpoint
from keystoneclient import exceptions
from keystoneclient import session
from keystoneclient.tests.unit.v3 import client_fixtures
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import client
def _management_url_is_updated(self, fixture, **kwargs):
    second = copy.deepcopy(fixture)
    first_url = 'http://admin:35357/v3'
    second_url = "http://secondurl:%d/v3'"
    for entry in second['token']['catalog']:
        if entry['type'] == 'identity':
            entry['endpoints'] = [{'url': second_url % 5000, 'region': 'RegionOne', 'interface': 'public'}, {'url': second_url % 5000, 'region': 'RegionOne', 'interface': 'internal'}, {'url': second_url % 35357, 'region': 'RegionOne', 'interface': 'admin'}]
    self.stub_auth(response_list=[{'json': fixture}, {'json': second}])
    with self.deprecations.expect_deprecations_here():
        cl = client.Client(username='exampleuser', password='password', auth_url=self.TEST_URL, **kwargs)
    self.assertEqual(cl.management_url, first_url)
    with self.deprecations.expect_deprecations_here():
        cl.authenticate()
    self.assertEqual(cl.management_url, second_url % 35357)