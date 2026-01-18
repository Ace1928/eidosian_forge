import logging
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import tags as _tag
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class SetSubnetPool(common.NeutronCommandWithExtraArgs):
    _description = _('Set subnet pool properties')

    def get_parser(self, prog_name):
        parser = super(SetSubnetPool, self).get_parser(prog_name)
        parser.add_argument('subnet_pool', metavar='<subnet-pool>', help=_('Subnet pool to modify (name or ID)'))
        parser.add_argument('--name', metavar='<name>', help=_('Set subnet pool name'))
        _add_prefix_options(parser)
        address_scope_group = parser.add_mutually_exclusive_group()
        address_scope_group.add_argument('--address-scope', metavar='<address-scope>', help=_('Set address scope associated with the subnet pool (name or ID), prefixes must be unique across address scopes'))
        address_scope_group.add_argument('--no-address-scope', action='store_true', help=_('Remove address scope associated with the subnet pool'))
        _add_default_options(parser)
        parser.add_argument('--description', metavar='<description>', help=_('Set subnet pool description'))
        (parser.add_argument('--default-quota', type=int, metavar='<num-ip-addresses>', help=_('Set default per-project quota for this subnet pool as the number of IP addresses that can be allocated from the subnet pool')),)
        _tag.add_tag_option_to_parser_for_set(parser, _('subnet pool'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        obj = client.find_subnet_pool(parsed_args.subnet_pool, ignore_missing=False)
        attrs = _get_attrs(self.app.client_manager, parsed_args)
        if 'prefixes' in attrs:
            attrs['prefixes'].extend(obj.prefixes)
        attrs.update(self._parse_extra_properties(parsed_args.extra_properties))
        if attrs:
            client.update_subnet_pool(obj, **attrs)
        _tag.update_tags_for_set(client, obj, parsed_args)