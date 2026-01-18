from cliff import lister
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import tags as _tag
from oslo_serialization import jsonutils
from oslo_utils import uuidutils
from octaviaclient.osc.v2 import constants as const
from octaviaclient.osc.v2 import utils as v2_utils
class ShowLoadBalancerStatus(command.Command):
    """Display load balancer status tree in json format"""

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('loadbalancer', metavar='<load_balancer>', help='Name or UUID of the load balancer.')
        return parser

    def take_action(self, parsed_args):
        attrs = v2_utils.get_loadbalancer_attrs(self.app.client_manager, parsed_args)
        lb_id = attrs.pop('loadbalancer_id')
        data = self.app.client_manager.load_balancer.load_balancer_status_show(lb_id=lb_id)
        res = data.get('statuses', {})
        print(jsonutils.dumps(res, indent=4))