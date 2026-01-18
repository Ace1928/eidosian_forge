import uuid
import testtools
from openstack import exceptions
from openstack.tests.unit import base
def _get_user_list(self, user_data):
    uri = self._get_keystone_mock_url(resource='users')
    return {'users': [user_data.json_response['user']], 'links': {'self': uri, 'previous': None, 'next': None}}