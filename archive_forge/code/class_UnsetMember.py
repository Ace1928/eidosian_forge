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
class UnsetMember(command.Command):
    """Clear member settings"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('pool', metavar='<pool>', help='Pool that the member to update belongs to (name or ID).')
        parser.add_argument('member', metavar='<member>', help='Member to modify (name or ID).')
        parser.add_argument('--backup', action='store_true', help='Clear the backup member flag.')
        parser.add_argument('--monitor-address', action='store_true', help='Clear the member monitor address.')
        parser.add_argument('--monitor-port', action='store_true', help='Clear the member monitor port.')
        parser.add_argument('--name', action='store_true', help='Clear the member name.')
        parser.add_argument('--weight', action='store_true', help='Reset the member weight to the API default.')
        parser.add_argument('--wait', action='store_true', help='Wait for action to complete.')
        _tag.add_tag_option_to_parser_for_unset(parser, 'member')
        return parser

    def take_action(self, parsed_args):
        unset_args = v2_utils.get_unsets(parsed_args)
        if not unset_args and (not parsed_args.all_tag):
            return
        pool_id = v2_utils.get_resource_id(self.app.client_manager.load_balancer.pool_list, 'pools', parsed_args.pool)
        member_show = functools.partial(self.app.client_manager.load_balancer.member_show, pool_id)
        member_dict = {'pool_id': pool_id, 'member_id': parsed_args.member}
        member_id = v2_utils.get_resource_id(self.app.client_manager.load_balancer.member_list, 'members', member_dict)
        v2_utils.set_tags_for_unset(member_show, member_id, unset_args, clear_tags=parsed_args.all_tag)
        body = {'member': unset_args}
        self.app.client_manager.load_balancer.member_set(pool_id=pool_id, member_id=member_id, json=body)
        if parsed_args.wait:
            v2_utils.wait_for_active(status_f=member_show, res_id=member_id)