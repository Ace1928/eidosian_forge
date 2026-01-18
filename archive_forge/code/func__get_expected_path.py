import requests
import uuid
from urllib import parse as urlparse
from keystoneauth1.identity import v3
from keystoneauth1 import session
from keystoneclient.tests.unit import client_fixtures
from keystoneclient.tests.unit import utils
from keystoneclient.v3 import client
def _get_expected_path(self, expected_path=None):
    if not expected_path:
        if self.path_prefix:
            expected_path = 'v3/%s/%s' % (self.path_prefix, self.collection_key)
        else:
            expected_path = 'v3/%s' % self.collection_key
    return expected_path