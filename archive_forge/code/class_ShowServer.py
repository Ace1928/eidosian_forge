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
class ShowServer(command.ShowOne):
    _description = _('Show server details.\n\nSpecify ``--os-compute-api-version 2.47`` or higher to see the embedded flavor\ninformation for the server.')

    def get_parser(self, prog_name):
        parser = super(ShowServer, self).get_parser(prog_name)
        parser.add_argument('server', metavar='<server>', help=_('Server (name or ID)'))
        diagnostics_group = parser.add_mutually_exclusive_group()
        diagnostics_group.add_argument('--diagnostics', action='store_true', default=False, help=_('Display server diagnostics information'))
        diagnostics_group.add_argument('--topology', action='store_true', default=False, help=_('Include topology information in the output (supported by --os-compute-api-version 2.78 or above)'))
        return parser

    def take_action(self, parsed_args):
        compute_client = self.app.client_manager.sdk_connection.compute
        server = compute_client.find_server(parsed_args.server, ignore_missing=False)
        server = compute_client.get_server(server)
        if parsed_args.diagnostics:
            data = compute_client.get_server_diagnostics(server)
            return zip(*sorted(data.items()))
        topology = None
        if parsed_args.topology:
            if not sdk_utils.supports_microversion(compute_client, '2.78'):
                msg = _('--os-compute-api-version 2.78 or greater is required to support the --topology option')
                raise exceptions.CommandError(msg)
            topology = server.fetch_topology(compute_client)
        data = _prep_server_detail(self.app.client_manager.compute, self.app.client_manager.image, server, refresh=False)
        if topology:
            data['topology'] = format_columns.DictColumn(topology)
        return zip(*sorted(data.items()))