from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import columns as column_util
from oslo_log import log as logging
from neutronclient._i18n import _
from neutronclient.common import utils as nc_utils
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.vpnaas import utils as vpn_utils
class SetIPsecSiteConnection(command.Command):
    _description = _('Set IPsec site connection properties')

    def get_parser(self, prog_name):
        parser = super(SetIPsecSiteConnection, self).get_parser(prog_name)
        _get_common_parser(parser)
        parser.add_argument('--peer-id', help=_('Peer router identity for authentication. Can be IPv4/IPv6 address, e-mail address, key id, or FQDN'))
        parser.add_argument('--peer-address', help=_('Peer gateway public IPv4/IPv6 address or FQDN'))
        parser.add_argument('--name', metavar='<name>', help=_('Set friendly name for the connection'))
        parser.add_argument('ipsec_site_connection', metavar='<ipsec-site-connection>', help=_('IPsec site connection to set (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        attrs = _get_common_attrs(self.app.client_manager, parsed_args, is_create=False)
        if parsed_args.peer_id:
            attrs['peer_id'] = parsed_args.peer_id
        if parsed_args.peer_address:
            attrs['peer_address'] = parsed_args.peer_address
        if parsed_args.name:
            attrs['name'] = parsed_args.name
        ipsec_conn_id = client.find_vpn_ipsec_site_connection(parsed_args.ipsec_site_connection, ignore_missing=False)['id']
        try:
            client.update_vpn_ipsec_site_connection(ipsec_conn_id, **attrs)
        except Exception as e:
            msg = _("Failed to set IPsec site connection '%(ipsec_conn)s': %(e)s") % {'ipsec_conn': parsed_args.ipsec_site_connection, 'e': e}
            raise exceptions.CommandError(msg)