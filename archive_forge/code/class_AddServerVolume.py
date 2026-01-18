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
class AddServerVolume(command.ShowOne):
    _description = _('Add volume to server.\n\nSpecify ``--os-compute-api-version 2.20`` or higher to add a volume to a server\nwith status ``SHELVED`` or ``SHELVED_OFFLOADED``.')

    def get_parser(self, prog_name):
        parser = super(AddServerVolume, self).get_parser(prog_name)
        parser.add_argument('server', metavar='<server>', help=_('Server (name or ID)'))
        parser.add_argument('volume', metavar='<volume>', help=_('Volume to add (name or ID)'))
        parser.add_argument('--device', metavar='<device>', help=_('Server internal device name for volume'))
        parser.add_argument('--tag', metavar='<tag>', help=_('Tag for the attached volume (supported by --os-compute-api-version 2.49 or above)'))
        termination_group = parser.add_mutually_exclusive_group()
        termination_group.add_argument('--enable-delete-on-termination', action='store_true', help=_('Delete the volume when the server is destroyed (supported by --os-compute-api-version 2.79 or above)'))
        termination_group.add_argument('--disable-delete-on-termination', action='store_true', help=_('Do not delete the volume when the server is destroyed (supported by --os-compute-api-version 2.79 or above)'))
        return parser

    def take_action(self, parsed_args):
        compute_client = self.app.client_manager.sdk_connection.compute
        volume_client = self.app.client_manager.sdk_connection.volume
        server = compute_client.find_server(parsed_args.server, ignore_missing=False)
        volume = volume_client.find_volume(parsed_args.volume, ignore_missing=False)
        kwargs = {'volumeId': volume.id, 'device': parsed_args.device}
        if parsed_args.tag:
            if not sdk_utils.supports_microversion(compute_client, '2.49'):
                msg = _('--os-compute-api-version 2.49 or greater is required to support the --tag option')
                raise exceptions.CommandError(msg)
            kwargs['tag'] = parsed_args.tag
        if parsed_args.enable_delete_on_termination:
            if not sdk_utils.supports_microversion(compute_client, '2.79'):
                msg = _('--os-compute-api-version 2.79 or greater is required to support the --enable-delete-on-termination option.')
                raise exceptions.CommandError(msg)
            kwargs['delete_on_termination'] = True
        if parsed_args.disable_delete_on_termination:
            if not sdk_utils.supports_microversion(compute_client, '2.79'):
                msg = _('--os-compute-api-version 2.79 or greater is required to support the --disable-delete-on-termination option.')
                raise exceptions.CommandError(msg)
            kwargs['delete_on_termination'] = False
        volume_attachment = compute_client.create_volume_attachment(server, **kwargs)
        columns = ('id', 'server id', 'volume id', 'device')
        column_headers = ('ID', 'Server ID', 'Volume ID', 'Device')
        if sdk_utils.supports_microversion(compute_client, '2.49'):
            columns += ('tag',)
            column_headers += ('Tag',)
        if sdk_utils.supports_microversion(compute_client, '2.79'):
            columns += ('delete_on_termination',)
            column_headers += ('Delete On Termination',)
        return (column_headers, utils.get_item_properties(volume_attachment, columns))