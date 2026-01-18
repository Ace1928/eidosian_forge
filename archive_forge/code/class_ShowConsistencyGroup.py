import argparse
import logging
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ShowConsistencyGroup(command.ShowOne):
    _description = _('Display consistency group details.')

    def get_parser(self, prog_name):
        parser = super(ShowConsistencyGroup, self).get_parser(prog_name)
        parser.add_argument('consistency_group', metavar='<consistency-group>', help=_('Consistency group to display (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        consistency_group = utils.find_resource(volume_client.consistencygroups, parsed_args.consistency_group)
        return zip(*sorted(consistency_group._info.items()))