import argparse
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.neutron.v2_0 import dns
from neutronclient.neutron.v2_0.qos import policy as qos_policy
class ListRouterPort(neutronV20.ListCommand):
    """List ports that belong to a given tenant, with specified router."""
    resource = 'port'
    _formatters = {'fixed_ips': _format_fixed_ips}
    list_columns = ['id', 'name', 'mac_address', 'fixed_ips']
    pagination_support = True
    sorting_support = True

    def get_parser(self, prog_name):
        parser = super(ListRouterPort, self).get_parser(prog_name)
        parser.add_argument('id', metavar='ROUTER', help=_('ID or name of the router to look up.'))
        return parser

    def take_action(self, parsed_args):
        neutron_client = self.get_client()
        _id = neutronV20.find_resourceid_by_name_or_id(neutron_client, 'router', parsed_args.id)
        self.values_specs.append('--device_id=%s' % _id)
        return super(ListRouterPort, self).take_action(parsed_args)