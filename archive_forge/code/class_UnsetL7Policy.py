from cliff import lister
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import tags as _tag
from oslo_utils import uuidutils
from octaviaclient.osc.v2 import constants as const
from octaviaclient.osc.v2 import utils as v2_utils
from octaviaclient.osc.v2 import validate
class UnsetL7Policy(command.Command):
    """Clear l7policy settings"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('l7policy', metavar='<policy>', help='L7policy to update (name or ID).')
        parser.add_argument('--description', action='store_true', help='Clear the l7policy description.')
        parser.add_argument('--name', action='store_true', help='Clear the l7policy name.')
        parser.add_argument('--redirect-http-code', action='store_true', help='Clear the l7policy redirect HTTP code.')
        parser.add_argument('--wait', action='store_true', help='Wait for action to complete.')
        _tag.add_tag_option_to_parser_for_unset(parser, 'l7policy')
        return parser

    def take_action(self, parsed_args):
        unset_args = v2_utils.get_unsets(parsed_args)
        if not unset_args and (not parsed_args.all_tag):
            return
        policy_id = v2_utils.get_resource_id(self.app.client_manager.load_balancer.l7policy_list, 'l7policies', parsed_args.l7policy)
        v2_utils.set_tags_for_unset(self.app.client_manager.load_balancer.l7policy_show, policy_id, unset_args, clear_tags=parsed_args.all_tag)
        body = {'l7policy': unset_args}
        self.app.client_manager.load_balancer.l7policy_set(policy_id, json=body)
        if parsed_args.wait:
            v2_utils.wait_for_active(status_f=self.app.client_manager.load_balancer.l7policy_show, res_id=policy_id)