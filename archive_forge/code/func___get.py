import testtools
from testtools import matchers
from openstack import exceptions
from openstack.tests.unit import base
def __get(self, resource, data, attr, qs_elements, use_name=False, is_found=True):
    if not use_name:
        if is_found:
            return [dict(method='GET', uri=self.get_mock_url(resource=resource + 's', append=[getattr(data, attr)], qs_elements=qs_elements), status_code=200, json=data.json_response)]
        else:
            return [dict(method='GET', uri=self.get_mock_url(resource=resource + 's', append=[getattr(data, attr)], qs_elements=qs_elements), status_code=404), dict(method='GET', uri=self.get_mock_url(resource=resource + 's', qs_elements=qs_elements), status_code=200, json={resource + 's': []})]
    else:
        return [dict(method='GET', uri=self.get_mock_url(resource=resource + 's', append=[getattr(data, attr)], qs_elements=qs_elements), status_code=404), dict(method='GET', uri=self.get_mock_url(resource=resource + 's', qs_elements=['name=' + getattr(data, attr)] + qs_elements), status_code=200, json={resource + 's': [data.json_response[resource]]})]