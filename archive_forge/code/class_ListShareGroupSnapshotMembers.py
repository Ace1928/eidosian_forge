import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib import utils as osc_utils
from manilaclient.common._i18n import _
class ListShareGroupSnapshotMembers(command.Lister):
    """List members for share group snapshot."""
    _description = _('List members of share group snapshot')

    def get_parser(self, prog_name):
        parser = super(ListShareGroupSnapshotMembers, self).get_parser(prog_name)
        parser.add_argument('share_group_snapshot', metavar='<share-group-snapshot>', help=_('Name or ID of the group snapshot to list members for'))
        return parser

    def take_action(self, parsed_args):
        share_client = self.app.client_manager.share
        columns = ['Share ID', 'Size']
        share_group_snapshot = osc_utils.find_resource(share_client.share_group_snapshots, parsed_args.share_group_snapshot)
        data = (osc_utils.get_dict_properties(member, columns) for member in share_group_snapshot._info.get('members', []))
        return (columns, data)