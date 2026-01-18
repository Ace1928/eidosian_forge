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
class MigrateVolume(command.Command):
    _description = _('Migrate volume to a new host')

    def get_parser(self, prog_name):
        parser = super(MigrateVolume, self).get_parser(prog_name)
        parser.add_argument('volume', metavar='<volume>', help=_('Volume to migrate (name or ID)'))
        parser.add_argument('--host', metavar='<host>', required=True, help=_('Destination host (takes the form: host@backend-name#pool)'))
        parser.add_argument('--force-host-copy', action='store_true', help=_('Enable generic host-based force-migration, which bypasses driver optimizations'))
        parser.add_argument('--lock-volume', action='store_true', help=_('If specified, the volume state will be locked and will not allow a migration to be aborted (possibly by another operation)'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.volume
        volume = utils.find_resource(volume_client.volumes, parsed_args.volume)
        volume_client.volumes.migrate_volume(volume.id, parsed_args.host, parsed_args.force_host_copy, parsed_args.lock_volume)