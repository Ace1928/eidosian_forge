import abc
import copy
from urllib import parse as urlparse
from ironicclient.common.apiclient import base
from ironicclient import exc
def __list(self, url, response_key=None, body=None, os_ironic_api_version=None, global_request_id=None):
    kwargs = {'headers': {}}
    if os_ironic_api_version is not None:
        kwargs['headers']['X-OpenStack-Ironic-API-Version'] = os_ironic_api_version
    if global_request_id is not None:
        kwargs['headers']['X-Openstack-Request-Id'] = global_request_id
    resp, body = self.api.json_request('GET', url, **kwargs)
    data = self._format_body_data(body, response_key)
    return data