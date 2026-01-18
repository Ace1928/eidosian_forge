import copy
import json
import logging
from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from osc_lib.utils import tags as _tag
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class AddExtraRoutesToRouter(command.ShowOne):
    _description = _("Add extra static routes to a router's routing table.")

    def get_parser(self, prog_name):
        parser = super(AddExtraRoutesToRouter, self).get_parser(prog_name)
        parser.add_argument('router', metavar='<router>', help=_('Router to which extra static routes will be added (name or ID).'))
        parser.add_argument('--route', metavar='destination=<subnet>,gateway=<ip-address>', action=parseractions.MultiKeyValueAction, dest='routes', default=[], required_keys=['destination', 'gateway'], help=_("Add extra static route to the router. destination: destination subnet (in CIDR notation), gateway: nexthop IP address. Repeat option to add multiple routes. Trying to add a route that's already present (exactly, including destination and nexthop) in the routing table is allowed and is considered a successful operation."))
        return parser

    def take_action(self, parsed_args):
        if parsed_args.routes is not None:
            for route in parsed_args.routes:
                route['nexthop'] = route.pop('gateway')
        client = self.app.client_manager.network
        router_obj = client.add_extra_routes_to_router(client.find_router(parsed_args.router, ignore_missing=False), body={'router': {'routes': parsed_args.routes}})
        display_columns, columns = _get_columns(router_obj)
        data = utils.get_item_properties(router_obj, columns, formatters=_formatters)
        return (display_columns, data)