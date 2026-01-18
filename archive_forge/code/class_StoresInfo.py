import argparse
from base64 import b64encode
import logging
import os
import sys
from cinderclient import api_versions
from openstack import exceptions as sdk_exceptions
from openstack.image import image_signer
from osc_lib.api import utils as api_utils
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.common import progressbar
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
class StoresInfo(command.Lister):
    _description = _('Get available backends (only valid with Multi-Backend support)')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('--detail', action='store_true', default=None, help=_('Shows details of stores (admin only) (requires --os-image-api-version 2.15 or later)'))
        return parser

    def take_action(self, parsed_args):
        image_client = self.app.client_manager.image
        try:
            columns = ('id', 'description', 'is_default')
            column_headers = ('ID', 'Description', 'Default')
            if parsed_args.detail:
                columns += ('properties',)
                column_headers += ('Properties',)
            data = list(image_client.stores(details=parsed_args.detail))
        except sdk_exceptions.ResourceNotFound:
            msg = _('Multi Backend support not enabled')
            raise exceptions.CommandError(msg)
        else:
            return (column_headers, (utils.get_item_properties(store, columns, formatters=_formatters) for store in data))