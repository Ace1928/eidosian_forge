from libcloud.compute.base import NodeLocation, VolumeSnapshot
from libcloud.compute.types import Provider, LibcloudError, VolumeSnapshotState
from libcloud.utils.iso8601 import parse_date
from libcloud.common.rackspace import AUTH_URL
from libcloud.compute.drivers.openstack import (
class RackspaceFirstGenNodeDriver(OpenStack_1_0_NodeDriver):
    name = 'Rackspace Cloud (First Gen)'
    website = 'http://www.rackspace.com'
    connectionCls = RackspaceFirstGenConnection
    type = Provider.RACKSPACE_FIRST_GEN
    api_name = 'rackspace'

    def __init__(self, key, secret=None, secure=True, host=None, port=None, region='us', **kwargs):
        """
        @inherits:  :class:`NodeDriver.__init__`

        :param region: Region ID which should be used
        :type region: ``str``
        """
        if region not in ['us', 'uk']:
            raise ValueError('Invalid region: %s' % region)
        super().__init__(key=key, secret=secret, secure=secure, host=host, port=port, region=region, **kwargs)

    def list_locations(self):
        """
        Lists available locations

        Locations cannot be set or retrieved via the API, but currently
        there are two locations, DFW and ORD.

        @inherits: :class:`OpenStack_1_0_NodeDriver.list_locations`
        """
        if self.region == 'us':
            locations = [NodeLocation(0, 'Rackspace DFW1/ORD1', 'US', self)]
        elif self.region == 'uk':
            locations = [NodeLocation(0, 'Rackspace UK London', 'UK', self)]
        return locations

    def _ex_connection_class_kwargs(self):
        kwargs = self.openstack_connection_kwargs()
        kwargs['region'] = self.region
        return kwargs