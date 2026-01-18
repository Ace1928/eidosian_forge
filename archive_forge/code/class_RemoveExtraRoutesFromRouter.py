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
class RemoveExtraRoutesFromRouter(command.ShowOne):
    _description = _("Remove extra static routes from a router's routing table.")

    def get_parser(self, prog_name):
        parser = super(RemoveExtraRoutesFromRouter, self).get_parser(prog_name)
        parser.add_argument('router', metavar='<router>', help=_('Router from which extra static routes will be removed (name or ID).'))
        parser.add_argument('--route', metavar='destination=<subnet>,gateway=<ip-address>', action=parseractions.MultiKeyValueAction, dest='routes', default=[], required_keys=['destination', 'gateway'], help=_("Remove extra static route from the router. destination: destination subnet (in CIDR notation), gateway: nexthop IP address. Repeat option to remove multiple routes. Trying to remove a route that's already missing (fully, including destination and nexthop) from the routing table is allowed and is considered a successful operation."))
        return parser

    def take_action(self, parsed_args):
        if parsed_args.routes is not None:
            for route in parsed_args.routes:
                route['nexthop'] = route.pop('gateway')
        client = self.app.client_manager.network
        router_obj = client.remove_extra_routes_from_router(client.find_router(parsed_args.router, ignore_missing=False), body={'router': {'routes': parsed_args.routes}})
        display_columns, columns = _get_columns(router_obj)
        data = utils.get_item_properties(router_obj, columns, formatters=_formatters)
        return (display_columns, data)