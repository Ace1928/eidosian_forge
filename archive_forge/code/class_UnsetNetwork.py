from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import utils
from osc_lib.utils import tags as _tag
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class UnsetNetwork(common.NeutronUnsetCommandWithExtraArgs):
    _description = _('Unset network properties')

    def get_parser(self, prog_name):
        parser = super(UnsetNetwork, self).get_parser(prog_name)
        parser.add_argument('network', metavar='<network>', help=_('Network to modify (name or ID)'))
        _tag.add_tag_option_to_parser_for_unset(parser, _('network'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        obj = client.find_network(parsed_args.network, ignore_missing=False)
        attrs = self._parse_extra_properties(parsed_args.extra_properties)
        if attrs:
            client.update_network(obj, **attrs)
        _tag.update_tags_for_unset(client, obj, parsed_args)