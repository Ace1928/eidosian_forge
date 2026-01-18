from libcloud.compute.base import NodeLocation, VolumeSnapshot
from libcloud.compute.types import Provider, LibcloudError, VolumeSnapshotState
from libcloud.utils.iso8601 import parse_date
from libcloud.common.rackspace import AUTH_URL
from libcloud.compute.drivers.openstack import (
class RackspaceNodeDriver(OpenStack_1_1_NodeDriver):
    name = 'Rackspace Cloud (Next Gen)'
    website = 'http://www.rackspace.com'
    connectionCls = RackspaceConnection
    type = Provider.RACKSPACE
    _networks_url_prefix = '/os-networksv2'

    def __init__(self, key, secret=None, secure=True, host=None, port=None, region='dfw', **kwargs):
        """
        @inherits:  :class:`NodeDriver.__init__`

        :param region: ID of the region which should be used.
        :type region: ``str``
        """
        valid_regions = ENDPOINT_ARGS_MAP.keys()
        if region not in valid_regions:
            raise ValueError('Invalid region: %s' % region)
        if region == 'lon':
            self.api_name = 'rackspacenovalon'
        elif region == 'syd':
            self.api_name = 'rackspacenovasyd'
        else:
            self.api_name = 'rackspacenovaus'
        super().__init__(key=key, secret=secret, secure=secure, host=host, port=port, region=region, **kwargs)

    def _to_snapshot(self, api_node):
        if 'snapshot' in api_node:
            api_node = api_node['snapshot']
        extra = {'volume_id': api_node['volumeId'], 'name': api_node['displayName'], 'created': api_node['createdAt'], 'description': api_node['displayDescription'], 'status': api_node['status']}
        state = self.SNAPSHOT_STATE_MAP.get(api_node['status'], VolumeSnapshotState.UNKNOWN)
        try:
            created_td = parse_date(api_node['createdAt'])
        except ValueError:
            created_td = None
        snapshot = VolumeSnapshot(id=api_node['id'], driver=self, size=api_node['size'], extra=extra, created=created_td, state=state, name=api_node['displayName'])
        return snapshot

    def _ex_connection_class_kwargs(self):
        endpoint_args = ENDPOINT_ARGS_MAP[self.region]
        kwargs = self.openstack_connection_kwargs()
        kwargs['region'] = self.region
        kwargs['get_endpoint_args'] = endpoint_args
        return kwargs