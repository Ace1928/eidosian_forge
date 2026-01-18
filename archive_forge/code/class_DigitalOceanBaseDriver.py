from libcloud.utils.py3 import httplib, parse_qs, urlparse
from libcloud.common.base import BaseDriver, JsonResponse, ConnectionKey
from libcloud.common.types import LibcloudError, InvalidCredsError
class DigitalOceanBaseDriver(BaseDriver):
    """
    DigitalOcean BaseDriver
    """
    name = 'DigitalOcean'
    website = 'https://www.digitalocean.com'

    def __new__(cls, key, secret=None, api_version='v2', **kwargs):
        if cls is DigitalOceanBaseDriver:
            if api_version == 'v1' or secret is not None:
                raise DigitalOcean_v1_Error()
            elif api_version == 'v2':
                cls = DigitalOcean_v2_BaseDriver
            else:
                raise NotImplementedError('Unsupported API version: %s' % api_version)
        return super().__new__(cls, **kwargs)

    def ex_account_info(self):
        raise NotImplementedError('ex_account_info not implemented for this driver')

    def ex_list_events(self):
        raise NotImplementedError('ex_list_events not implemented for this driver')

    def ex_get_event(self, event_id):
        raise NotImplementedError('ex_get_event not implemented for this driver')

    def _paginated_request(self, url, obj):
        raise NotImplementedError('_paginated_requests not implemented for this driver')