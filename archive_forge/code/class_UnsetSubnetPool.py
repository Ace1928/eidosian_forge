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
class UnsetSubnetPool(command.Command):
    _description = _('Unset subnet pool properties')

    def get_parser(self, prog_name):
        parser = super(UnsetSubnetPool, self).get_parser(prog_name)
        parser.add_argument('subnet_pool', metavar='<subnet-pool>', help=_('Subnet pool to modify (name or ID)'))
        _tag.add_tag_option_to_parser_for_unset(parser, _('subnet pool'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        obj = client.find_subnet_pool(parsed_args.subnet_pool, ignore_missing=False)
        _tag.update_tags_for_unset(client, obj, parsed_args)