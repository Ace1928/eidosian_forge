from unittest import mock
from oslo_serialization import jsonutils
import sys
from keystoneauth1 import fixture
import requests
class FakeClientManager(object):
    _api_version = {'image': '2'}

    def __init__(self):
        self.compute = None
        self.identity = None
        self.image = None
        self.object_store = None
        self.volume = None
        self.network = None
        self.session = None
        self.auth_ref = None
        self.auth_plugin_name = None
        self.network_endpoint_enabled = True

    def get_configuration(self):
        return {'auth': {'username': USERNAME, 'password': PASSWORD, 'token': AUTH_TOKEN}, 'region': REGION_NAME, 'identity_api_version': VERSION}

    def is_network_endpoint_enabled(self):
        return self.network_endpoint_enabled