import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common import cliutils
from manilaclient.osc import utils
class ShowShareReplica(command.ShowOne):
    """Show share replica."""
    _description = _('Show details about a replica')

    def get_parser(self, prog_name):
        parser = super(ShowShareReplica, self).get_parser(prog_name)
        parser.add_argument('replica', metavar='<replica>', help=_('ID of the share replica. Available only for microversion >= 2.47. '))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        replica = share_client.share_replicas.get(parsed_args.replica)
        replica_export_locations = share_client.share_replica_export_locations.list(share_replica=replica)
        replica._info['export_locations'] = []
        for element_location in replica_export_locations:
            element_location._info.pop('links', None)
            replica._info['export_locations'].append(element_location._info)
        if parsed_args.formatter == 'table':
            replica._info['export_locations'] = cliutils.convert_dict_list_to_string(replica._info['export_locations'])
        replica._info.pop('links', None)
        return self.dict2columns(replica._info)