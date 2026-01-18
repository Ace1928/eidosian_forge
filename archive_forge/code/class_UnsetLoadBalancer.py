from cliff import lister
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import tags as _tag
from oslo_serialization import jsonutils
from oslo_utils import uuidutils
from octaviaclient.osc.v2 import constants as const
from octaviaclient.osc.v2 import utils as v2_utils
class UnsetLoadBalancer(command.Command):
    """Clear load balancer settings"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('loadbalancer', metavar='<load_balancer>', help='Name or UUID of the load balancer to update.')
        parser.add_argument('--name', action='store_true', help='Clear the load balancer name.')
        parser.add_argument('--description', action='store_true', help='Clear the load balancer description.')
        parser.add_argument('--vip-qos-policy-id', action='store_true', help='Clear the load balancer QoS policy.')
        parser.add_argument('--wait', action='store_true', help='Wait for action to complete.')
        _tag.add_tag_option_to_parser_for_unset(parser, 'load balancer')
        return parser

    def take_action(self, parsed_args):
        unset_args = v2_utils.get_unsets(parsed_args)
        if not unset_args and (not parsed_args.all_tag):
            return
        lb_id = v2_utils.get_resource_id(self.app.client_manager.load_balancer.load_balancer_list, 'loadbalancers', parsed_args.loadbalancer)
        v2_utils.set_tags_for_unset(self.app.client_manager.load_balancer.load_balancer_show, lb_id, unset_args, clear_tags=parsed_args.all_tag)
        body = {'loadbalancer': unset_args}
        self.app.client_manager.load_balancer.load_balancer_set(lb_id, json=body)
        if parsed_args.wait:
            v2_utils.wait_for_active(status_f=self.app.client_manager.load_balancer.load_balancer_show, res_id=lb_id)