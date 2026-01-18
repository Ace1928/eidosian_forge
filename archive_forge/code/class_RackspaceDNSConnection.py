import copy
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.utils.misc import get_new_obj, merge_valid_keys
from libcloud.common.base import PollingConnection
from libcloud.common.types import LibcloudError
from libcloud.common.openstack import OpenStackDriverMixin
from libcloud.common.rackspace import AUTH_URL
from libcloud.common.exceptions import BaseHTTPError
from libcloud.compute.drivers.openstack import OpenStack_1_1_Response, OpenStack_1_1_Connection
class RackspaceDNSConnection(OpenStack_1_1_Connection, PollingConnection):
    """
    Rackspace DNS Connection class.
    """
    responseCls = RackspaceDNSResponse
    XML_NAMESPACE = None
    poll_interval = 2.5
    timeout = 30
    auth_url = AUTH_URL
    _auth_version = '2.0'

    def __init__(self, *args, **kwargs):
        self.region = kwargs.pop('region', None)
        super().__init__(*args, **kwargs)

    def get_poll_request_kwargs(self, response, context, request_kwargs):
        job_id = response.object['jobId']
        kwargs = {'action': '/status/%s' % job_id, 'params': {'showDetails': True}}
        return kwargs

    def has_completed(self, response):
        status = response.object['status']
        if status == 'ERROR':
            data = response.object['error']
            if 'code' and 'message' in data:
                message = '{} - {} ({})'.format(data['code'], data['message'], data['details'])
            else:
                message = data['message']
            raise LibcloudError(message, driver=self.driver)
        return status == 'COMPLETED'

    def get_endpoint(self):
        if '2.0' in self._auth_version:
            ep = self.service_catalog.get_endpoint(name='cloudDNS', service_type='rax:dns', region=None)
        else:
            raise LibcloudError('Auth version %s not supported' % self._auth_version)
        public_url = ep.url
        if self.region == 'us':
            public_url = public_url.replace('https://lon.dns.api', 'https://dns.api')
        if self.region == 'uk':
            public_url = public_url.replace('https://dns.api', 'https://lon.dns.api')
        return public_url