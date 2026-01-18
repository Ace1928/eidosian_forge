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
class SetMember(command.Command):
    """Update a member"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('pool', metavar='<pool>', help='Pool that the member to update belongs to (name or ID).')
        parser.add_argument('member', metavar='<member>', help='Name or ID of the member to update.')
        parser.add_argument('--name', metavar='<name>', help='Set the name of the member.')
        backup = parser.add_mutually_exclusive_group()
        backup.add_argument('--disable-backup', action='store_true', default=None, help='Disable member backup (default).')
        backup.add_argument('--enable-backup', action='store_true', default=None, help='Enable member backup.')
        parser.add_argument('--weight', metavar='<weight>', type=int, help='Set the weight of member in the pool.')
        parser.add_argument('--monitor-port', metavar='<monitor_port>', type=int, help='An alternate protocol port used for health monitoring a backend member.')
        parser.add_argument('--monitor-address', metavar='<monitor_address>', help='An alternate IP address used for health monitoring a backend member.')
        admin_group = parser.add_mutually_exclusive_group()
        admin_group.add_argument('--enable', action='store_true', default=None, help='Set the admin_state_up to True.')
        admin_group.add_argument('--disable', action='store_true', default=None, help='Set the admin_state_up to False.')
        parser.add_argument('--wait', action='store_true', help='Wait for action to complete.')
        _tag.add_tag_option_to_parser_for_set(parser, 'member')
        return parser

    def take_action(self, parsed_args):
        attrs = v2_utils.get_member_attrs(self.app.client_manager, parsed_args)
        validate.check_member_attrs(attrs)
        pool_id = attrs.pop('pool_id')
        member_id = attrs.pop('member_id')
        member_show = functools.partial(self.app.client_manager.load_balancer.member_show, pool_id)
        v2_utils.set_tags_for_set(member_show, member_id, attrs, clear_tags=parsed_args.no_tag)
        post_data = {'member': attrs}
        self.app.client_manager.load_balancer.member_set(pool_id=pool_id, member_id=member_id, json=post_data)
        if parsed_args.wait:
            v2_utils.wait_for_active(status_f=member_show, res_id=member_id)