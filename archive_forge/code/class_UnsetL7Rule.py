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
class UnsetL7Rule(command.Command):
    """Clear l7rule settings"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('l7policy', metavar='<l7policy>', help='L7policy to update (name or ID).')
        parser.add_argument('l7rule_id', metavar='<l7rule_id>', help='l7rule to update.')
        parser.add_argument('--invert', action='store_true', help='Reset the l7rule invert to the API default.')
        parser.add_argument('--key', action='store_true', help='Clear the l7rule key.')
        parser.add_argument('--wait', action='store_true', help='Wait for action to complete.')
        _tag.add_tag_option_to_parser_for_unset(parser, 'l7rule')
        return parser

    def take_action(self, parsed_args):
        unset_args = v2_utils.get_unsets(parsed_args)
        if not unset_args and (not parsed_args.all_tag):
            return
        policy_id = v2_utils.get_resource_id(self.app.client_manager.load_balancer.l7policy_list, 'l7policies', parsed_args.l7policy)
        l7rule_show = functools.partial(self.app.client_manager.load_balancer.l7rule_show, parsed_args.l7rule_id)
        v2_utils.set_tags_for_unset(l7rule_show, policy_id, unset_args, clear_tags=parsed_args.all_tag)
        body = {'rule': unset_args}
        self.app.client_manager.load_balancer.l7rule_set(l7policy_id=policy_id, l7rule_id=parsed_args.l7rule_id, json=body)
        if parsed_args.wait:
            v2_utils.wait_for_active(status_f=l7rule_show, res_id=policy_id)