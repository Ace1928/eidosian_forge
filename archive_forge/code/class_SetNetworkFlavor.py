import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class SetNetworkFlavor(common.NeutronCommandWithExtraArgs):
    _description = _('Set network flavor properties')

    def get_parser(self, prog_name):
        parser = super(SetNetworkFlavor, self).get_parser(prog_name)
        parser.add_argument('flavor', metavar='<flavor>', help=_('Flavor to update (name or ID)'))
        parser.add_argument('--description', help=_('Set network flavor description'))
        enable_group = parser.add_mutually_exclusive_group()
        enable_group.add_argument('--disable', action='store_true', help=_('Disable network flavor'))
        enable_group.add_argument('--enable', action='store_true', help=_('Enable network flavor'))
        parser.add_argument('--name', metavar='<name>', help=_('Set flavor name'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        obj = client.find_flavor(parsed_args.flavor, ignore_missing=False)
        attrs = {}
        if parsed_args.name is not None:
            attrs['name'] = parsed_args.name
        if parsed_args.description is not None:
            attrs['description'] = parsed_args.description
        if parsed_args.enable:
            attrs['enabled'] = True
        if parsed_args.disable:
            attrs['enabled'] = False
        attrs.update(self._parse_extra_properties(parsed_args.extra_properties))
        client.update_flavor(obj, **attrs)