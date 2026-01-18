import argparse
import getpass
import io
import json
import logging
import os
from cliff import columns as cliff_columns
import iso8601
from novaclient import api_versions
from openstack import exceptions as sdk_exceptions
from openstack import utils as sdk_utils
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common as network_common
class RebootServer(command.Command):
    _description = _('Perform a hard or soft server reboot')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('server', metavar='<server>', help=_('Server (name or ID)'))
        group = parser.add_mutually_exclusive_group()
        group.add_argument('--hard', dest='reboot_type', action='store_const', const='HARD', default='SOFT', help=_('Perform a hard reboot'))
        group.add_argument('--soft', dest='reboot_type', action='store_const', const='SOFT', default='SOFT', help=_('Perform a soft reboot'))
        parser.add_argument('--wait', action='store_true', help=_('Wait for reboot to complete'))
        return parser

    def take_action(self, parsed_args):

        def _show_progress(progress):
            if progress:
                self.app.stdout.write('\rProgress: %s' % progress)
                self.app.stdout.flush()
        compute_client = self.app.client_manager.sdk_connection.compute
        server_id = compute_client.find_server(parsed_args.server, ignore_missing=False).id
        compute_client.reboot_server(server_id, parsed_args.reboot_type)
        if parsed_args.wait:
            if utils.wait_for_status(compute_client.get_server, server_id, callback=_show_progress):
                self.app.stdout.write(_('Complete\n'))
            else:
                msg = _('Error rebooting server: %s') % server_id
                raise exceptions.CommandError(msg)