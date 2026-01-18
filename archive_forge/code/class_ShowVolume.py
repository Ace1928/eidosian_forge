import argparse
import copy
import functools
import logging
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
class ShowVolume(command.ShowOne):
    _description = _('Display volume details')

    def get_parser(self, prog_name):
        parser = super(ShowVolume, self).get_parser(prog_name)
        parser.add_argument('volume', metavar='<volume>', help=_('Volume to display (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        volume = utils.find_resource(volume_client.volumes, parsed_args.volume)
        volume._info.update({'properties': format_columns.DictColumn(volume._info.pop('metadata')), 'type': volume._info.pop('volume_type')})
        volume._info.pop('links', None)
        return zip(*sorted(volume._info.items()))