import ipaddress
import time
import warnings
from openstack.cloud import _utils
from openstack.cloud import exc
from openstack.cloud import meta
from openstack import exceptions
from openstack.network.v2._proxy import Proxy
from openstack import proxy
from openstack import utils
from openstack import warnings as os_warnings
def _nat_destination_port(self, server, fixed_address=None, nat_destination=None):
    """Returns server port that is on a nat_destination network

        Find a port attached to the server which is on a network which
        has a subnet which can be the destination of NAT. Such a network
        is referred to in shade as a "nat_destination" network. So this
        then is a function which returns a port on such a network that is
        associated with the given server.

        :param server: Server dict.
        :param fixed_address: Fixed ip address of the port
        :param nat_destination: Name or ID of the network of the port.
        """
    port_filter = {'device_id': server['id']}
    ports = self.search_ports(filters=port_filter)
    if not ports:
        return (None, None)
    port = None
    if not fixed_address:
        if len(ports) > 1:
            if nat_destination:
                nat_network = self.get_network(nat_destination)
                if not nat_network:
                    raise exceptions.SDKException('NAT Destination {nat_destination} was configured but not found on the cloud. Please check your config and your cloud and try again.'.format(nat_destination=nat_destination))
            else:
                nat_network = self.get_nat_destination()
            if not nat_network:
                raise exceptions.SDKException('Multiple ports were found for server {server} but none of the networks are a valid NAT destination, so it is impossible to add a floating IP. If you have a network that is a valid destination for NAT and we could not find it, please file a bug. But also configure the nat_destination property of the networks list in your clouds.yaml file. If you do not have a clouds.yaml file, please make one - your setup is complicated.'.format(server=server['id']))
            maybe_ports = []
            for maybe_port in ports:
                if maybe_port['network_id'] == nat_network['id']:
                    maybe_ports.append(maybe_port)
            if not maybe_ports:
                raise exceptions.SDKException('No port on server {server} was found matching your NAT destination network {dest}. Please  check your config'.format(server=server['id'], dest=nat_network['name']))
            ports = maybe_ports
        for port in sorted(ports, key=lambda p: p.get('created_at', 0), reverse=True):
            for address in port.get('fixed_ips', list()):
                try:
                    ip = ipaddress.ip_address(address['ip_address'])
                except Exception:
                    continue
                if ip.version == 4:
                    fixed_address = address['ip_address']
                    return (port, fixed_address)
        raise exceptions.SDKException('unable to find a free fixed IPv4 address for server {0}'.format(server['id']))
    for p in ports:
        for fixed_ip in p['fixed_ips']:
            if fixed_address == fixed_ip['ip_address']:
                return (p, fixed_address)
    return (None, None)