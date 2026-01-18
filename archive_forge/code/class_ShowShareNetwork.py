import logging
from openstackclient.identity import common as identity_common
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common import cliutils
class ShowShareNetwork(command.ShowOne):
    """Display a share network"""
    _description = _('Show details about a share network')

    def get_parser(self, prog_name):
        parser = super(ShowShareNetwork, self).get_parser(prog_name)
        parser.add_argument('share_network', metavar='<share-network>', help=_('Name or ID of the share network to display'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share_network = oscutils.find_resource(share_client.share_networks, parsed_args.share_network)
        data = share_network._info
        for ss in data['share_network_subnets']:
            ss.update({'properties': format_columns.DictColumn(ss.pop('metadata', {}))})
        security_services = share_client.security_services.list(search_opts={'share_network_id': share_network.id}, detailed=False)
        data['security_services'] = [{'security_service_name': ss.name, 'security_service_id': ss.id} for ss in security_services]
        if parsed_args.formatter == 'table':
            data['share_network_subnets'] = cliutils.convert_dict_list_to_string(data['share_network_subnets'])
            data['security_services'] = cliutils.convert_dict_list_to_string(data['security_services'])
        data.pop('links', None)
        return self.dict2columns(data)