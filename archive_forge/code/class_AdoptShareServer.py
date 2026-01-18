import logging
from openstackclient.identity import common as identity_common
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.common import cliutils
from manilaclient.common import constants
class AdoptShareServer(command.ShowOne):
    """Adopt share server not handled by Manila (Admin only)."""
    _description = _('Adopt share server not handled by Manila (Admin only).')

    def get_parser(self, prog_name):
        parser = super(AdoptShareServer, self).get_parser(prog_name)
        parser.add_argument('host', metavar='<host>', type=str, help=_('Backend name as "<node_hostname>@<backend_name>".'))
        parser.add_argument('share_network', metavar='<share-network>', help=_('Share network where share server has network allocations in.'))
        parser.add_argument('identifier', metavar='<identifier>', type=str, help=_('A driver-specific share server identifier required by the driver to manage the share server.'))
        parser.add_argument('--driver-options', metavar='<key=value>', action=parseractions.KeyValueAction, default={}, help=_('One or more driver-specific key=value pairs that may be necessary to manage the share server (Optional, Default=None).'))
        parser.add_argument('--share-network-subnet', type=str, metavar='<share-network-subnet>', default=None, help="Share network subnet where share server has network  allocations in.The default subnet will be used if it's not specified. Available for microversion >= 2.51 (Optional, Default=None).")
        parser.add_argument('--wait', action='store_true', help=_('Wait until share server is adopted'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share_network = None
        if parsed_args.share_network:
            share_network = osc_utils.find_resource(share_client.share_networks, parsed_args.share_network).id
        share_network_subnet = None
        if parsed_args.share_network_subnet and share_client.api_version < api_versions.APIVersion('2.51'):
            raise exceptions.CommandError('Share network subnet can be specified only with manila API version >= 2.51')
        elif parsed_args.share_network_subnet:
            share_network_subnet = share_client.share_network_subnets.get(share_network, parsed_args.share_network_subnet).id
        share_server = share_client.share_servers.manage(host=parsed_args.host, share_network_id=share_network, identifier=parsed_args.identifier, driver_options=parsed_args.driver_options, share_network_subnet_id=share_network_subnet)
        if parsed_args.wait:
            if not osc_utils.wait_for_status(status_f=share_client.share_servers.get, res_id=share_server.id, success_status=['active'], error_status=['manage_error', 'error']):
                LOG.error(_('ERROR: Share server is in error state.'))
            share_server = osc_utils.find_resource(share_client.share_servers, share_server.id)
        share_server._info.pop('links', None)
        share_server._info.pop('backend_details', None)
        return self.dict2columns(share_server._info)