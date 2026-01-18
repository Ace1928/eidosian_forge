import copy
import json
import logging
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import tags as _tag
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class AddPortToRouter(command.Command):
    _description = _('Add a port to a router')

    def get_parser(self, prog_name):
        parser = super(AddPortToRouter, self).get_parser(prog_name)
        parser.add_argument('router', metavar='<router>', help=_('Router to which port will be added (name or ID)'))
        parser.add_argument('port', metavar='<port>', help=_('Port to be added (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        port = client.find_port(parsed_args.port, ignore_missing=False)
        client.add_interface_to_router(client.find_router(parsed_args.router, ignore_missing=False), port_id=port.id)