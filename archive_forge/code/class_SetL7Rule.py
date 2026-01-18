import functools
from cliff import lister
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import tags as _tag
from oslo_utils import uuidutils
from octaviaclient.osc.v2 import constants as const
from octaviaclient.osc.v2 import utils as v2_utils
from octaviaclient.osc.v2 import validate
class SetL7Rule(command.Command):
    """Update a l7rule"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('l7policy', metavar='<l7policy>', help='L7policy to update l7rule on (name or ID).')
        parser.add_argument('l7rule', metavar='<l7rule_id>', help='l7rule to update.')
        parser.add_argument('--compare-type', metavar='{' + ','.join(COMPARE_TYPES) + '}', choices=COMPARE_TYPES, type=lambda s: s.upper(), help='Set the compare type for the l7rule.')
        parser.add_argument('--invert', action='store_true', default=None, help='Invert l7rule.')
        parser.add_argument('--value', metavar='<value>', help='Set the rule value to match on.')
        parser.add_argument('--key', metavar='<key>', help="Set the key for the l7rule's value to match on.")
        parser.add_argument('--type', metavar='{' + ','.join(TYPES) + '}', choices=TYPES, type=lambda s: s.upper(), help='Set the type for the l7rule.')
        admin_group = parser.add_mutually_exclusive_group()
        admin_group.add_argument('--enable', action='store_true', default=None, help='Enable l7rule.')
        admin_group.add_argument('--disable', action='store_true', default=None, help='Disable l7rule.')
        parser.add_argument('--wait', action='store_true', help='Wait for action to complete.')
        _tag.add_tag_option_to_parser_for_set(parser, 'l7rule')
        return parser

    def take_action(self, parsed_args):
        attrs = v2_utils.get_l7rule_attrs(self.app.client_manager, parsed_args)
        validate.check_l7rule_attrs(attrs)
        l7policy_id = attrs.pop('l7policy_id')
        l7rule_id = attrs.pop('l7rule_id')
        l7rule_show = functools.partial(self.app.client_manager.load_balancer.l7rule_show, l7rule_id)
        v2_utils.set_tags_for_set(l7rule_show, l7policy_id, attrs, clear_tags=parsed_args.no_tag)
        body = {'rule': attrs}
        self.app.client_manager.load_balancer.l7rule_set(l7rule_id=l7rule_id, l7policy_id=l7policy_id, json=body)
        if parsed_args.wait:
            v2_utils.wait_for_active(status_f=l7rule_show, res_id=l7policy_id)