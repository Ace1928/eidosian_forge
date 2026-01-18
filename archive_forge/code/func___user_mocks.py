import testtools
from testtools import matchers
from openstack import exceptions
from openstack.tests.unit import base
def __user_mocks(self, user_data, use_name, is_found=True):
    uri_mocks = []
    if not use_name:
        uri_mocks.append(dict(method='GET', uri=self.get_mock_url(resource='users'), status_code=200, json={'users': [user_data.json_response['user']] if is_found else []}))
    else:
        uri_mocks.append(dict(method='GET', uri=self.get_mock_url(resource='users', qs_elements=['name=' + user_data.name]), status_code=200, json={'users': [user_data.json_response['user']] if is_found else []}))
    return uri_mocks