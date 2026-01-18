import logging
from osc_lib.cli import format_columns
from osc_lib.cli.parseractions import KeyValueAction
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from osc_lib.utils import columns as column_util
from neutronclient._i18n import _
from neutronclient.osc import utils as nc_osc_utils
class DeleteBgpvpn(command.Command):
    _description = _('Delete BGP VPN resource(s)')

    def get_parser(self, prog_name):
        parser = super(DeleteBgpvpn, self).get_parser(prog_name)
        parser.add_argument('bgpvpns', metavar='<bgpvpn>', nargs='+', help=_('BGP VPN(s) to delete (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        fails = 0
        for id_or_name in parsed_args.bgpvpns:
            try:
                id = client.find_bgpvpn(id_or_name)['id']
                client.delete_bgpvpn(id)
                LOG.warning('BGP VPN %(id)s deleted', {'id': id})
            except Exception as e:
                fails += 1
                LOG.error("Failed to delete BGP VPN with name or ID '%(id_or_name)s': %(e)s", {'id_or_name': id_or_name, 'e': e})
        if fails > 0:
            msg = _('Failed to delete %(fails)s of %(total)s BGP VPN.') % {'fails': fails, 'total': len(parsed_args.bgpvpns)}
            raise exceptions.CommandError(msg)