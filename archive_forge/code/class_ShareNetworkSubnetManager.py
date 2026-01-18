from manilaclient import api_versions
from manilaclient import base
class ShareNetworkSubnetManager(base.MetadataCapableManager):
    """Manage :class:`ShareNetworkSubnet` resources."""
    resource_class = ShareNetworkSubnet
    resource_path = '/share-networks'
    subresource_path = '/subnets'

    def _do_create(self, neutron_net_id=None, neutron_subnet_id=None, availability_zone=None, share_network_id=None, metadata=None):
        """Create share network subnet.

        :param neutron_net_id: ID of Neutron network
        :param neutron_subnet_id: ID of Neutron subnet
        :param availability_zone: Name of the target availability zone
        :param metadata: dict - optional metadata to set on share creation
        :rtype: :class:`ShareNetworkSubnet`
        """
        values = {}
        if neutron_net_id:
            values['neutron_net_id'] = neutron_net_id
        if neutron_subnet_id:
            values['neutron_subnet_id'] = neutron_subnet_id
        if availability_zone:
            values['availability_zone'] = availability_zone
        if metadata:
            values['metadata'] = metadata
        body = {'share-network-subnet': values}
        url = '/share-networks/%(share_network_id)s/subnets' % {'share_network_id': share_network_id}
        return self._create(url, body, RESOURCE_NAME)

    @api_versions.wraps('2.0', '2.77')
    def create(self, neutron_net_id=None, neutron_subnet_id=None, availability_zone=None, share_network_id=None):
        return self._do_create(neutron_net_id, neutron_subnet_id, availability_zone, share_network_id)

    @api_versions.wraps('2.78')
    def create(self, neutron_net_id=None, neutron_subnet_id=None, availability_zone=None, share_network_id=None, metadata=None):
        return self._do_create(neutron_net_id, neutron_subnet_id, availability_zone, share_network_id, metadata)

    def get(self, share_network, share_network_subnet):
        """Get a share network subnet.

        :param policy: share network subnet to get.
        :rtype: :class:`NetworkSubnetInfo`
        """
        share_network_id = base.getid(share_network)
        share_network_subnet_id = base.getid(share_network_subnet)
        url = '/share-networks/%(share_network_id)s/subnets/%(share_network_subnet)s' % {'share_network_id': share_network_id, 'share_network_subnet': share_network_subnet_id}
        return self._get(url, 'share_network_subnet')

    def delete(self, share_network, share_network_subnet):
        """Delete a share network subnet.

        :param share_network: share network that owns the subnet.
        :param share_network_subnet: share network subnet to be deleted.
        """
        url = '/share-networks/%(share_network_id)s/subnets/%(share_network_subnet)s' % {'share_network_id': base.getid(share_network), 'share_network_subnet': share_network_subnet}
        self._delete(url)

    @api_versions.wraps('2.78')
    def get_metadata(self, share_network, share_network_subnet):
        return super(ShareNetworkSubnetManager, self).get_metadata(share_network, subresource=share_network_subnet)

    @api_versions.wraps('2.78')
    def set_metadata(self, resource, metadata, subresource=None):
        return super(ShareNetworkSubnetManager, self).set_metadata(resource, metadata, subresource=subresource)

    @api_versions.wraps('2.78')
    def delete_metadata(self, resource, keys, subresource=None):
        return super(ShareNetworkSubnetManager, self).delete_metadata(resource, keys, subresource=subresource)

    @api_versions.wraps('2.78')
    def update_all_metadata(self, resource, metadata, subresource=None):
        return super(ShareNetworkSubnetManager, self).update_all_metadata(resource, metadata, subresource=subresource)