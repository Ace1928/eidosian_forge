import argparse
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class CreateSubnet(neutronV20.CreateCommand):
    """Create a subnet for a given tenant."""
    resource = 'subnet'

    def add_known_arguments(self, parser):
        add_updatable_arguments(parser)
        parser.add_argument('--ip-version', type=int, default=4, choices=[4, 6], help=_('IP version to use, default is 4. Note that when subnetpool is specified, IP version is determined from the subnetpool and this option is ignored.'))
        parser.add_argument('--ip_version', type=int, choices=[4, 6], help=argparse.SUPPRESS)
        parser.add_argument('network_id', metavar='NETWORK', help=_('Network ID or name this subnet belongs to.'))
        parser.add_argument('cidr', nargs='?', metavar='CIDR', help=_('CIDR of subnet to create.'))
        parser.add_argument('--ipv6-ra-mode', type=utils.convert_to_lowercase, choices=['dhcpv6-stateful', 'dhcpv6-stateless', 'slaac'], help=_('IPv6 RA (Router Advertisement) mode.'))
        parser.add_argument('--ipv6-address-mode', type=utils.convert_to_lowercase, choices=['dhcpv6-stateful', 'dhcpv6-stateless', 'slaac'], help=_('IPv6 address mode.'))
        parser.add_argument('--subnetpool', metavar='SUBNETPOOL', help=_('ID or name of subnetpool from which this subnet will obtain a CIDR.'))
        parser.add_argument('--use-default-subnetpool', action='store_true', help=_('Use default subnetpool for ip_version, if it exists.'))
        parser.add_argument('--prefixlen', metavar='PREFIX_LENGTH', help=_('Prefix length for subnet allocation from subnetpool.'))
        parser.add_argument('--segment', metavar='SEGMENT', help=_('ID of segment with which this subnet will be associated.'))

    def args2body(self, parsed_args):
        _network_id = neutronV20.find_resourceid_by_name_or_id(self.get_client(), 'network', parsed_args.network_id)
        body = {'network_id': _network_id}
        if parsed_args.prefixlen:
            body['prefixlen'] = parsed_args.prefixlen
        ip_version = parsed_args.ip_version
        if parsed_args.use_default_subnetpool:
            body['use_default_subnetpool'] = True
        if parsed_args.segment:
            body['segment_id'] = neutronV20.find_resourceid_by_name_or_id(self.get_client(), 'segment', parsed_args.segment)
        if parsed_args.subnetpool:
            if parsed_args.subnetpool == 'None':
                _subnetpool_id = None
            else:
                _subnetpool = neutronV20.find_resource_by_name_or_id(self.get_client(), 'subnetpool', parsed_args.subnetpool)
                _subnetpool_id = _subnetpool['id']
                ip_version = _subnetpool['ip_version']
            body['subnetpool_id'] = _subnetpool_id
        body['ip_version'] = ip_version
        if parsed_args.cidr:
            cidr = parsed_args.cidr
            body['cidr'] = cidr
            unusable_cidr = '/32' if ip_version == 4 else '/128'
            if cidr.endswith(unusable_cidr):
                self.log.warning(_('An IPv%(ip)d subnet with a %(cidr)s CIDR will have only one usable IP address so the device attached to it will not have any IP connectivity.'), {'ip': ip_version, 'cidr': unusable_cidr})
        updatable_args2body(parsed_args, body, ip_version=ip_version)
        if parsed_args.tenant_id:
            body['tenant_id'] = parsed_args.tenant_id
        return {'subnet': body}