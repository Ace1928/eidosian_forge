import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class SetAddressScope(common.NeutronCommandWithExtraArgs):
    _description = _('Set address scope properties')

    def get_parser(self, prog_name):
        parser = super(SetAddressScope, self).get_parser(prog_name)
        parser.add_argument('address_scope', metavar='<address-scope>', help=_('Address scope to modify (name or ID)'))
        parser.add_argument('--name', metavar='<name>', help=_('Set address scope name'))
        share_group = parser.add_mutually_exclusive_group()
        share_group.add_argument('--share', action='store_true', help=_('Share the address scope between projects'))
        share_group.add_argument('--no-share', action='store_true', help=_('Do not share the address scope between projects'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        obj = client.find_address_scope(parsed_args.address_scope, ignore_missing=False)
        attrs = {}
        if parsed_args.name is not None:
            attrs['name'] = parsed_args.name
        if parsed_args.share:
            attrs['shared'] = True
        if parsed_args.no_share:
            attrs['shared'] = False
        attrs.update(self._parse_extra_properties(parsed_args.extra_properties))
        client.update_address_scope(obj, **attrs)