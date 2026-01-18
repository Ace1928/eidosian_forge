import argparse
from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.neutron.v2_0 import dns
class DisassociateFloatingIP(neutronV20.NeutronCommand):
    """Remove a mapping from a floating IP to a fixed IP."""
    resource = 'floatingip'

    def get_parser(self, prog_name):
        parser = super(DisassociateFloatingIP, self).get_parser(prog_name)
        parser.add_argument('floatingip_id', metavar='FLOATINGIP_ID', help=_('ID of the floating IP to disassociate.'))
        return parser

    def take_action(self, parsed_args):
        neutron_client = self.get_client()
        neutron_client.update_floatingip(parsed_args.floatingip_id, {'floatingip': {'port_id': None}})
        print(_('Disassociated floating IP %s') % parsed_args.floatingip_id, file=self.app.stdout)