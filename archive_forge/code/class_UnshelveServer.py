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
class UnshelveServer(command.Command):
    _description = _('Unshelve server(s)')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('server', metavar='<server>', nargs='+', help=_('Server(s) to unshelve (name or ID)'))
        group = parser.add_mutually_exclusive_group()
        group.add_argument('--availability-zone', default=None, help=_('Name of the availability zone in which to unshelve a SHELVED_OFFLOADED server (supported by --os-compute-api-version 2.77 or above)'))
        group.add_argument('--no-availability-zone', action='store_true', default=False, help=_('Unpin the availability zone of a SHELVED_OFFLOADED server. Server will be unshelved on a host without availability zone constraint (supported by --os-compute-api-version 2.91 or above)'))
        parser.add_argument('--host', default=None, help=_('Name of the destination host in which to unshelve a SHELVED_OFFLOADED server (supported by --os-compute-api-version 2.91 or above)'))
        parser.add_argument('--wait', action='store_true', default=False, help=_('Wait for unshelve operation to complete'))
        return parser

    def take_action(self, parsed_args):

        def _show_progress(progress):
            if progress:
                self.app.stdout.write('\rProgress: %s' % progress)
                self.app.stdout.flush()
        compute_client = self.app.client_manager.sdk_connection.compute
        kwargs = {}
        if parsed_args.availability_zone:
            if not sdk_utils.supports_microversion(compute_client, '2.77'):
                msg = _('--os-compute-api-version 2.77 or greater is required to support the --availability-zone option')
                raise exceptions.CommandError(msg)
            kwargs['availability_zone'] = parsed_args.availability_zone
        if parsed_args.host:
            if not sdk_utils.supports_microversion(compute_client, '2.91'):
                msg = _('--os-compute-api-version 2.91 or greater is required to support the --host option')
                raise exceptions.CommandError(msg)
            kwargs['host'] = parsed_args.host
        if parsed_args.no_availability_zone:
            if not sdk_utils.supports_microversion(compute_client, '2.91'):
                msg = _('--os-compute-api-version 2.91 or greater is required to support the --no-availability-zone option')
                raise exceptions.CommandError(msg)
            kwargs['availability_zone'] = None
        for server in parsed_args.server:
            server_obj = compute_client.find_server(server, ignore_missing=False)
            if server_obj.status.lower() not in ('shelved', 'shelved_offloaded'):
                continue
            compute_client.unshelve_server(server_obj.id, **kwargs)
            if parsed_args.wait:
                if not utils.wait_for_status(compute_client.get_server, server_obj.id, success_status=('active', 'shutoff'), callback=_show_progress):
                    msg = _('Error unshelving server: %s') % server_obj.id
                    raise exceptions.CommandError(msg)