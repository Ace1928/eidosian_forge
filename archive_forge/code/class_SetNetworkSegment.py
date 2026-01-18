import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.network import common
class SetNetworkSegment(common.NeutronCommandWithExtraArgs):
    _description = _('Set network segment properties')

    def get_parser(self, prog_name):
        parser = super(SetNetworkSegment, self).get_parser(prog_name)
        parser.add_argument('network_segment', metavar='<network-segment>', help=_('Network segment to modify (name or ID)'))
        parser.add_argument('--description', metavar='<description>', help=_('Set network segment description'))
        parser.add_argument('--name', metavar='<name>', help=_('Set network segment name'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        obj = client.find_segment(parsed_args.network_segment, ignore_missing=False)
        attrs = {}
        if parsed_args.description is not None:
            attrs['description'] = parsed_args.description
        if parsed_args.name is not None:
            attrs['name'] = parsed_args.name
        attrs.update(self._parse_extra_properties(parsed_args.extra_properties))
        client.update_segment(obj, **attrs)