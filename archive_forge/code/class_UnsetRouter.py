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
class UnsetRouter(common.NeutronUnsetCommandWithExtraArgs):
    _description = _('Unset router properties')

    def get_parser(self, prog_name):
        parser = super(UnsetRouter, self).get_parser(prog_name)
        parser.add_argument('--route', metavar='destination=<subnet>,gateway=<ip-address>', action=parseractions.MultiKeyValueAction, dest='routes', default=None, required_keys=['destination', 'gateway'], help=_('Routes to be removed from the router destination: destination subnet (in CIDR notation) gateway: nexthop IP address (repeat option to unset multiple routes)'))
        parser.add_argument('--external-gateway', action='store_true', default=False, help=_('Remove external gateway information from the router'))
        parser.add_argument('--qos-policy', action='store_true', default=False, help=_('Remove QoS policy from router gateway IPs'))
        parser.add_argument('router', metavar='<router>', help=_('Router to modify (name or ID)'))
        _tag.add_tag_option_to_parser_for_unset(parser, _('router'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        obj = client.find_router(parsed_args.router, ignore_missing=False)
        tmp_routes = copy.deepcopy(obj.routes)
        tmp_external_gateway_info = copy.deepcopy(obj.external_gateway_info)
        attrs = {}
        if parsed_args.routes:
            try:
                for route in parsed_args.routes:
                    route['nexthop'] = route.pop('gateway')
                    tmp_routes.remove(route)
            except ValueError:
                msg = _('Router does not contain route %s') % route
                raise exceptions.CommandError(msg)
            attrs['routes'] = tmp_routes
        if parsed_args.qos_policy:
            try:
                if tmp_external_gateway_info['network_id'] and tmp_external_gateway_info['qos_policy_id']:
                    pass
            except (KeyError, TypeError):
                msg = _('Router does not have external network or qos policy')
                raise exceptions.CommandError(msg)
            else:
                attrs['external_gateway_info'] = {'network_id': tmp_external_gateway_info['network_id'], 'qos_policy_id': None}
        if parsed_args.external_gateway:
            attrs['external_gateway_info'] = {}
        attrs.update(self._parse_extra_properties(parsed_args.extra_properties))
        if attrs:
            client.update_router(obj, **attrs)
        _tag.update_tags_for_unset(client, obj, parsed_args)