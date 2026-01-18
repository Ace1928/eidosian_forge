import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ShowConsistencyGroupSnapshot(command.ShowOne):
    _description = _('Display consistency group snapshot details')

    def get_parser(self, prog_name):
        parser = super(ShowConsistencyGroupSnapshot, self).get_parser(prog_name)
        parser.add_argument('consistency_group_snapshot', metavar='<consistency-group-snapshot>', help=_('Consistency group snapshot to display (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        consistency_group_snapshot = utils.find_resource(volume_client.cgsnapshots, parsed_args.consistency_group_snapshot)
        return zip(*sorted(consistency_group_snapshot._info.items()))