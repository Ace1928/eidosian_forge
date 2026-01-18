from libcloud.compute.base import NodeLocation, VolumeSnapshot
from libcloud.compute.types import Provider, LibcloudError, VolumeSnapshotState
from libcloud.utils.iso8601 import parse_date
from libcloud.common.rackspace import AUTH_URL
from libcloud.compute.drivers.openstack import (
class RackspaceFirstGenConnection(OpenStack_1_0_Connection):
    """
    Connection class for the Rackspace first-gen driver.
    """
    responseCls = OpenStack_1_0_Response
    XML_NAMESPACE = 'http://docs.rackspacecloud.com/servers/api/v1.0'
    auth_url = AUTH_URL
    _auth_version = '2.0'
    cache_busting = True

    def __init__(self, *args, **kwargs):
        self.region = kwargs.pop('region', None)
        super().__init__(*args, **kwargs)

    def get_endpoint(self):
        if '2.0' in self._auth_version:
            ep = self.service_catalog.get_endpoint(service_type=SERVICE_TYPE, name=SERVICE_NAME_GEN1)
        else:
            raise LibcloudError('Auth version "%s" not supported' % self._auth_version)
        public_url = ep.url
        if not public_url:
            raise LibcloudError('Could not find specified endpoint')
        if self.region == 'us':
            public_url = public_url.replace('https://lon.servers.api', 'https://servers.api')
        elif self.region == 'uk':
            public_url = public_url.replace('https://servers.api', 'https://lon.servers.api')
        return public_url

    def get_service_name(self):
        return SERVICE_NAME_GEN1