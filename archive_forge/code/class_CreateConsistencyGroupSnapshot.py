import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class CreateConsistencyGroupSnapshot(command.ShowOne):
    _description = _('Create new consistency group snapshot.')

    def get_parser(self, prog_name):
        parser = super(CreateConsistencyGroupSnapshot, self).get_parser(prog_name)
        parser.add_argument('snapshot_name', metavar='<snapshot-name>', nargs='?', help=_('Name of new consistency group snapshot (default to None)'))
        parser.add_argument('--consistency-group', metavar='<consistency-group>', help=_('Consistency group to snapshot (name or ID) (default to be the same as <snapshot-name>)'))
        parser.add_argument('--description', metavar='<description>', help=_('Description of this consistency group snapshot'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        consistency_group = parsed_args.consistency_group
        if not parsed_args.consistency_group:
            consistency_group = parsed_args.snapshot_name
        consistency_group_id = utils.find_resource(volume_client.consistencygroups, consistency_group).id
        consistency_group_snapshot = volume_client.cgsnapshots.create(consistency_group_id, name=parsed_args.snapshot_name, description=parsed_args.description)
        return zip(*sorted(consistency_group_snapshot._info.items()))