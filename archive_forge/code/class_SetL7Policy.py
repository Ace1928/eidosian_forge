from cliff import lister
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import tags as _tag
from oslo_utils import uuidutils
from octaviaclient.osc.v2 import constants as const
from octaviaclient.osc.v2 import utils as v2_utils
from octaviaclient.osc.v2 import validate
class SetL7Policy(command.Command):
    """Update a l7policy"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('l7policy', metavar='<policy>', help='L7policy to update (name or ID).')
        parser.add_argument('--name', metavar='<name>', help='Set l7policy name.')
        parser.add_argument('--description', metavar='<description>', help='Set l7policy description.')
        parser.add_argument('--action', metavar='{' + ','.join(ACTION_CHOICES) + '}', choices=ACTION_CHOICES, type=lambda s: s.upper(), help='Set the action of the policy.')
        redirect_group = parser.add_mutually_exclusive_group()
        redirect_group.add_argument('--redirect-pool', metavar='<pool>', help='Set the pool to redirect requests to (name or ID).')
        redirect_group.add_argument('--redirect-url', metavar='<url>', help='Set the URL to redirect requests to.')
        redirect_group.add_argument('--redirect-prefix', metavar='<url>', help='Set the URL Prefix to redirect requests to.')
        parser.add_argument('--redirect-http-code', metavar='<redirect_http_code>', choices=REDIRECT_CODE_CHOICES, type=int, help='Set the HTTP response code for REDIRECT_URL or REDIRECT_PREFIX action.')
        parser.add_argument('--position', metavar='<position>', type=int, help='Set sequence number of this L7 Policy.')
        admin_group = parser.add_mutually_exclusive_group()
        admin_group.add_argument('--enable', action='store_true', default=None, help='Enable l7policy.')
        admin_group.add_argument('--disable', action='store_true', default=None, help='Disable l7policy.')
        parser.add_argument('--wait', action='store_true', help='Wait for action to complete.')
        _tag.add_tag_option_to_parser_for_set(parser, 'l7policy')
        return parser

    def take_action(self, parsed_args):
        attrs = v2_utils.get_l7policy_attrs(self.app.client_manager, parsed_args)
        validate.check_l7policy_attrs(attrs)
        l7policy_id = attrs.pop('l7policy_id')
        v2_utils.set_tags_for_set(self.app.client_manager.load_balancer.l7policy_show, l7policy_id, attrs, clear_tags=parsed_args.no_tag)
        body = {'l7policy': attrs}
        self.app.client_manager.load_balancer.l7policy_set(l7policy_id, json=body)
        if parsed_args.wait:
            v2_utils.wait_for_active(status_f=self.app.client_manager.load_balancer.l7policy_show, res_id=l7policy_id)