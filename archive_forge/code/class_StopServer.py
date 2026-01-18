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
class StopServer(command.Command):
    _description = _('Stop server(s)')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('server', metavar='<server>', nargs='+', help=_('Server(s) to stop (name or ID)'))
        parser.add_argument('--all-projects', action='store_true', default=boolenv('ALL_PROJECTS'), help=_('Stop server(s) in another project by name (admin only)(can be specified using the ALL_PROJECTS envvar)'))
        return parser

    def take_action(self, parsed_args):
        compute_client = self.app.client_manager.sdk_connection.compute
        for server in parsed_args.server:
            try:
                server_id = compute_client.find_server(server, ignore_missing=False, details=False, all_projects=parsed_args.all_projects).id
            except sdk_exceptions.HttpException as exc:
                if exc.status_code == 403:
                    msg = _("Policy doesn't allow passing all-projects")
                    raise exceptions.Forbidden(msg)
            compute_client.stop_server(server_id)