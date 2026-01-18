import argparse
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.neutron.v2_0 import availability_zone
class UpdateRouter(neutronV20.UpdateCommand):
    """Update router's information."""
    resource = 'router'

    def add_known_arguments(self, parser):
        parser.add_argument('--name', help=_('Updated name of the router.'))
        parser.add_argument('--description', help=_('Description of router.'))
        utils.add_boolean_argument(parser, '--admin-state-up', dest='admin_state', help=_('Specify the administrative state of the router (True means "Up").'))
        utils.add_boolean_argument(parser, '--admin_state_up', dest='admin_state', help=argparse.SUPPRESS)
        utils.add_boolean_argument(parser, '--distributed', dest='distributed', help=_('True means this router should operate in distributed mode.'))
        routes_group = parser.add_mutually_exclusive_group()
        routes_group.add_argument('--route', metavar='destination=CIDR,nexthop=IP_ADDR', action='append', dest='routes', type=utils.str2dict_type(required_keys=['destination', 'nexthop']), help=_('Route to associate with the router. You can repeat this option.'))
        routes_group.add_argument('--no-routes', action='store_true', help=_('Remove routes associated with the router.'))

    def args2body(self, parsed_args):
        body = {}
        if hasattr(parsed_args, 'admin_state'):
            body['admin_state_up'] = parsed_args.admin_state
        neutronV20.update_dict(parsed_args, body, ['name', 'distributed', 'description'])
        if parsed_args.no_routes:
            body['routes'] = None
        elif parsed_args.routes:
            body['routes'] = parsed_args.routes
        return {self.resource: body}