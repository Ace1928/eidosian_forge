import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
class SetNDPProxy(command.Command):
    _description = _('Set NDP proxy properties')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('ndp_proxy', metavar='<ndp-proxy>', help=_('The ID or name of the NDP proxy to update'))
        parser.add_argument('--name', metavar='<name>', help=_('Set NDP proxy name'))
        parser.add_argument('--description', metavar='<description>', help=_('A text to describe/contextualize the use of the NDP proxy configuration'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        attrs = {}
        if parsed_args.description is not None:
            attrs['description'] = parsed_args.description
        if parsed_args.name is not None:
            attrs['name'] = parsed_args.name
        obj = client.find_ndp_proxy(parsed_args.ndp_proxy, ignore_missing=False)
        client.update_ndp_proxy(obj, **attrs)