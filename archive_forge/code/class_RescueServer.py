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
class RescueServer(command.Command):
    _description = _('Put server in rescue mode.\n\nSpecify ``--os-compute-api-version 2.87`` or higher to rescue a\nserver booted from a volume.')

    def get_parser(self, prog_name):
        parser = super(RescueServer, self).get_parser(prog_name)
        parser.add_argument('server', metavar='<server>', help=_('Server (name or ID)'))
        parser.add_argument('--image', metavar='<image>', help=_('Image (name or ID) to use for the rescue mode. Defaults to the currently used one.'))
        parser.add_argument('--password', metavar='<password>', help=_('Set the password on the rescued instance. This option requires cloud support.'))
        return parser

    def take_action(self, parsed_args):
        compute_client = self.app.client_manager.compute
        image_client = self.app.client_manager.image
        image = None
        if parsed_args.image:
            image = image_client.find_image(parsed_args.image, ignore_missing=False)
        utils.find_resource(compute_client.servers, parsed_args.server).rescue(image=image, password=parsed_args.password)