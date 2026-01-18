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
class ResizeConfirm(command.Command):
    _description = _('Confirm server resize.\n\nConfirm (verify) success of resize operation and release the old server.')

    def get_parser(self, prog_name):
        parser = super(ResizeConfirm, self).get_parser(prog_name)
        parser.add_argument('server', metavar='<server>', help=_('Server (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        compute_client = self.app.client_manager.compute
        server = utils.find_resource(compute_client.servers, parsed_args.server)
        server.confirm_resize()