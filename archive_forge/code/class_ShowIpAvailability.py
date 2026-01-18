from cliff import show
from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronV20
class ShowIpAvailability(neutronV20.NeutronCommand, show.ShowOne):
    """Show IP usage of specific network"""
    resource = 'network_ip_availability'

    def get_parser(self, prog_name):
        parser = super(ShowIpAvailability, self).get_parser(prog_name)
        parser.add_argument('network_id', metavar='NETWORK', help=_('ID or name of network to look up.'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('run(%s)', parsed_args)
        neutron_client = self.get_client()
        _id = neutronV20.find_resourceid_by_name_or_id(neutron_client, 'network', parsed_args.network_id)
        data = neutron_client.show_network_ip_availability(_id)
        self.format_output_data(data)
        resource = data[self.resource]
        if self.resource in data:
            return zip(*sorted(resource.items()))
        else:
            return None