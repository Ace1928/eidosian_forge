import threading
from openstack import exceptions
def _set_interesting_networks(self):
    external_ipv4_networks = []
    external_ipv4_floating_networks = []
    internal_ipv4_networks = []
    external_ipv6_networks = []
    internal_ipv6_networks = []
    nat_destination = None
    nat_source = None
    default_network = None
    all_subnets = None
    try:
        all_networks = self.list_networks()
    except exceptions.SDKException:
        self._network_list_stamp = True
        return
    for network in all_networks:
        if network['name'] in self._external_ipv4_names or network['id'] in self._external_ipv4_names:
            external_ipv4_networks.append(network)
        elif (network.is_router_external or network.provider_physical_network) and network['name'] not in self._internal_ipv4_names and (network['id'] not in self._internal_ipv4_names):
            external_ipv4_networks.append(network)
        if network['name'] in self._internal_ipv4_names or network['id'] in self._internal_ipv4_names:
            internal_ipv4_networks.append(network)
        elif not network.is_router_external and (not network.provider_physical_network) and (network['name'] not in self._external_ipv4_names) and (network['id'] not in self._external_ipv4_names):
            internal_ipv4_networks.append(network)
        if network['name'] in self._external_ipv6_names or network['id'] in self._external_ipv6_names:
            external_ipv6_networks.append(network)
        elif network.is_router_external and network['name'] not in self._internal_ipv6_names and (network['id'] not in self._internal_ipv6_names):
            external_ipv6_networks.append(network)
        if network['name'] in self._internal_ipv6_names or network['id'] in self._internal_ipv6_names:
            internal_ipv6_networks.append(network)
        elif not network.is_router_external and network['name'] not in self._external_ipv6_names and (network['id'] not in self._external_ipv6_names):
            internal_ipv6_networks.append(network)
        if self._nat_source in (network['name'], network['id']):
            if nat_source:
                raise exceptions.SDKException('Multiple networks were found matching {nat_net} which is the network configured to be the NAT source. Please check your cloud resources. It is probably a good idea to configure this network by ID rather than by name.'.format(nat_net=self._nat_source))
            external_ipv4_floating_networks.append(network)
            nat_source = network
        elif self._nat_source is None:
            if network.is_router_external:
                external_ipv4_floating_networks.append(network)
                nat_source = nat_source or network
        if self._nat_destination in (network['name'], network['id']):
            if nat_destination:
                raise exceptions.SDKException('Multiple networks were found matching {nat_net} which is the network configured to be the NAT destination. Please check your cloud resources. It is probably a good idea to configure this network by ID rather than by name.'.format(nat_net=self._nat_destination))
            nat_destination = network
        elif self._nat_destination is None:
            if all_subnets is None:
                try:
                    all_subnets = self.list_subnets()
                except exceptions.SDKException:
                    all_subnets = []
            for subnet in all_subnets:
                if 'gateway_ip' in subnet and subnet['gateway_ip'] and (network['id'] == subnet['network_id']):
                    nat_destination = network
                    break
        if self._default_network in (network['name'], network['id']):
            if default_network:
                raise exceptions.SDKException('Multiple networks were found matching {default_net} which is the network configured to be the default interface network. Please check your cloud resources. It is probably a good idea to configure this network by ID rather than by name.'.format(default_net=self._default_network))
            default_network = network
    for net_name in self._external_ipv4_names:
        if net_name not in [net['name'] for net in external_ipv4_networks]:
            raise exceptions.SDKException('Networks: {network} was provided for external IPv4 access and those networks could not be found'.format(network=net_name))
    for net_name in self._internal_ipv4_names:
        if net_name not in [net['name'] for net in internal_ipv4_networks]:
            raise exceptions.SDKException('Networks: {network} was provided for internal IPv4 access and those networks could not be found'.format(network=net_name))
    for net_name in self._external_ipv6_names:
        if net_name not in [net['name'] for net in external_ipv6_networks]:
            raise exceptions.SDKException('Networks: {network} was provided for external IPv6 access and those networks could not be found'.format(network=net_name))
    for net_name in self._internal_ipv6_names:
        if net_name not in [net['name'] for net in internal_ipv6_networks]:
            raise exceptions.SDKException('Networks: {network} was provided for internal IPv6 access and those networks could not be found'.format(network=net_name))
    if self._nat_destination and (not nat_destination):
        raise exceptions.SDKException('Network {network} was configured to be the destination for inbound NAT but it could not be found'.format(network=self._nat_destination))
    if self._nat_source and (not nat_source):
        raise exceptions.SDKException('Network {network} was configured to be the source for inbound NAT but it could not be found'.format(network=self._nat_source))
    if self._default_network and (not default_network):
        raise exceptions.SDKException('Network {network} was configured to be the default network interface but it could not be found'.format(network=self._default_network))
    self._external_ipv4_networks = external_ipv4_networks
    self._external_ipv4_floating_networks = external_ipv4_floating_networks
    self._internal_ipv4_networks = internal_ipv4_networks
    self._external_ipv6_networks = external_ipv6_networks
    self._internal_ipv6_networks = internal_ipv6_networks
    self._nat_destination_network = nat_destination
    self._nat_source_network = nat_source
    self._default_network_network = default_network