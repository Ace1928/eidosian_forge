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
class CreateServerDump(command.Command):
    """Create a dump file in server(s)

    Trigger crash dump in server(s) with features like kdump in Linux.
    It will create a dump file in the server(s) dumping the server(s)'
    memory, and also crash the server(s). This is contingent on guest operating
    system support, and the location of the dump file inside the guest will
    depend on the exact guest operating system.

    This command requires ``--os-compute-api-version`` 2.17 or greater.
    """

    def get_parser(self, prog_name):
        parser = super(CreateServerDump, self).get_parser(prog_name)
        parser.add_argument('server', metavar='<server>', nargs='+', help=_('Server(s) to create dump file (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        compute_client = self.app.client_manager.sdk_connection.compute
        for name_or_id in parsed_args.server:
            server = compute_client.find_server(name_or_id)
            server.trigger_crash_dump(compute_client)