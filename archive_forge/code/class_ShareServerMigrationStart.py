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
class ShareServerMigrationStart(command.ShowOne):
    """Migrates share server to a new host (Admin only, Experimental)."""
    _description = _('Migrates share server to a new host.')

    def get_parser(self, prog_name):
        parser = super(ShareServerMigrationStart, self).get_parser(prog_name)
        parser.add_argument('share_server', metavar='<share_server>', help=_('ID of share server to start migration.'))
        parser.add_argument('host', metavar='<host@backend>', help=_("Destination to migrate the share server to. Use the format '<node_hostname>@<backend_name>'."))
        parser.add_argument('--preserve-snapshots', metavar='<True|False>', choices=['True', 'False'], required=True, help=_('Set to True if snapshots must be preserved at the migration destination.'))
        parser.add_argument('--writable', metavar='<True|False>', choices=['True', 'False'], required=True, help=_('Enforces migration to keep all its shares writable while contents are being moved.'))
        parser.add_argument('--nondisruptive', metavar='<True|False>', choices=['True', 'False'], required=True, help=_('Enforces migration to be nondisruptive.'))
        parser.add_argument('--new-share-network', metavar='<new_share_network>', required=False, default=None, help=_('Specify a new share network for the share server. Do not specify this parameter if the migrating share server has to be retained within its current share network.'))
        parser.add_argument('--check-only', action='store_true', default=False, help=_('Run a dry-run of the share server migration. '))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        share_server = osc_utils.find_resource(share_client.share_servers, parsed_args.share_server)
        if share_client.api_version >= api_versions.APIVersion('2.57'):
            new_share_net_id = None
            result = None
            if parsed_args.new_share_network:
                new_share_net_id = apiutils.find_resource(share_client.share_networks, parsed_args.new_share_network).id
            if parsed_args.check_only:
                result = share_server.migration_check(parsed_args.host, parsed_args.writable, parsed_args.nondisruptive, parsed_args.preserve_snapshots, new_share_net_id)
            if result:
                if parsed_args.formatter == 'table':
                    for k, v in result.items():
                        if isinstance(v, dict):
                            capabilities_list = [v]
                            dict_values = cliutils.convert_dict_list_to_string(capabilities_list)
                            result[k] = dict_values
                return self.dict2columns(result)
            else:
                share_server.migration_start(parsed_args.host, parsed_args.writable, parsed_args.nondisruptive, parsed_args.preserve_snapshots, new_share_net_id)
                return ({}, {})
        else:
            raise exceptions.CommandError('Share Server Migration is only available with manila API version >= 2.57')