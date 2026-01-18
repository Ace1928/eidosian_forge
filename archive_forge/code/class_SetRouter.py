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
class SetRouter(common.NeutronCommandWithExtraArgs):
    _description = _('Set router properties')

    def get_parser(self, prog_name):
        parser = super(SetRouter, self).get_parser(prog_name)
        parser.add_argument('router', metavar='<router>', help=_('Router to modify (name or ID)'))
        parser.add_argument('--name', metavar='<name>', help=_('Set router name'))
        parser.add_argument('--description', metavar='<description>', help=_('Set router description'))
        admin_group = parser.add_mutually_exclusive_group()
        admin_group.add_argument('--enable', action='store_true', default=None, help=_('Enable router'))
        admin_group.add_argument('--disable', action='store_true', help=_('Disable router'))
        distribute_group = parser.add_mutually_exclusive_group()
        distribute_group.add_argument('--distributed', action='store_true', help=_('Set router to distributed mode (disabled router only)'))
        distribute_group.add_argument('--centralized', action='store_true', help=_('Set router to centralized mode (disabled router only)'))
        parser.add_argument('--route', metavar='destination=<subnet>,gateway=<ip-address>', action=parseractions.MultiKeyValueAction, dest='routes', default=None, required_keys=['destination', 'gateway'], help=_("Add routes to the router destination: destination subnet (in CIDR notation) gateway: nexthop IP address (repeat option to add multiple routes). This is deprecated in favor of 'router add/remove route' since it is prone to race conditions between concurrent clients when not used together with --no-route to overwrite the current value of 'routes'."))
        parser.add_argument('--no-route', action='store_true', help=_('Clear routes associated with the router. Specify both --route and --no-route to overwrite current value of routes.'))
        routes_ha = parser.add_mutually_exclusive_group()
        routes_ha.add_argument('--ha', action='store_true', help=_('Set the router as highly available (disabled router only)'))
        routes_ha.add_argument('--no-ha', action='store_true', help=_('Clear high availability attribute of the router (disabled router only)'))
        parser.add_argument('--external-gateway', metavar='<network>', help=_("External Network used as router's gateway (name or ID)"))
        parser.add_argument('--fixed-ip', metavar='subnet=<subnet>,ip-address=<ip-address>', action=parseractions.MultiKeyValueAction, optional_keys=['subnet', 'ip-address'], help=_('Desired IP and/or subnet (name or ID) on external gateway: subnet=<subnet>,ip-address=<ip-address> (repeat option to set multiple fixed IP addresses)'))
        snat_group = parser.add_mutually_exclusive_group()
        snat_group.add_argument('--enable-snat', action='store_true', help=_('Enable Source NAT on external gateway'))
        snat_group.add_argument('--disable-snat', action='store_true', help=_('Disable Source NAT on external gateway'))
        ndp_proxy_group = parser.add_mutually_exclusive_group()
        ndp_proxy_group.add_argument('--enable-ndp-proxy', dest='enable_ndp_proxy', default=None, action='store_true', help=_('Enable IPv6 NDP proxy on external gateway'))
        ndp_proxy_group.add_argument('--disable-ndp-proxy', dest='enable_ndp_proxy', default=None, action='store_false', help=_('Disable IPv6 NDP proxy on external gateway'))
        qos_policy_group = parser.add_mutually_exclusive_group()
        qos_policy_group.add_argument('--qos-policy', metavar='<qos-policy>', help=_('Attach QoS policy to router gateway IPs'))
        qos_policy_group.add_argument('--no-qos-policy', action='store_true', help=_('Remove QoS policy from router gateway IPs'))
        _tag.add_tag_option_to_parser_for_set(parser, _('router'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        obj = client.find_router(parsed_args.router, ignore_missing=False)
        attrs = _get_attrs(self.app.client_manager, parsed_args)
        if parsed_args.ha:
            attrs['ha'] = True
        elif parsed_args.no_ha:
            attrs['ha'] = False
        if parsed_args.routes is not None:
            for route in parsed_args.routes:
                route['nexthop'] = route.pop('gateway')
            attrs['routes'] = parsed_args.routes
            if not parsed_args.no_route:
                attrs['routes'] += obj.routes
        elif parsed_args.no_route:
            attrs['routes'] = []
        if (parsed_args.disable_snat or parsed_args.enable_snat or parsed_args.fixed_ip) and (not parsed_args.external_gateway):
            msg = _("You must specify '--external-gateway' in order to update the SNAT or fixed-ip values")
            raise exceptions.CommandError(msg)
        if (parsed_args.qos_policy or parsed_args.no_qos_policy) and (not parsed_args.external_gateway):
            try:
                original_net_id = obj.external_gateway_info['network_id']
            except (KeyError, TypeError):
                msg = _("You must specify '--external-gateway' or the router must already have an external network in order to set router gateway IP QoS")
                raise exceptions.CommandError(msg)
            else:
                if not attrs.get('external_gateway_info'):
                    attrs['external_gateway_info'] = {}
                attrs['external_gateway_info']['network_id'] = original_net_id
        if parsed_args.qos_policy:
            check_qos_id = client.find_qos_policy(parsed_args.qos_policy, ignore_missing=False).id
            attrs['external_gateway_info']['qos_policy_id'] = check_qos_id
        if 'no_qos_policy' in parsed_args and parsed_args.no_qos_policy:
            attrs['external_gateway_info']['qos_policy_id'] = None
        attrs.update(self._parse_extra_properties(parsed_args.extra_properties))
        if parsed_args.enable_ndp_proxy is not None:
            attrs['enable_ndp_proxy'] = parsed_args.enable_ndp_proxy
        if attrs:
            client.update_router(obj, **attrs)
        _tag.update_tags_for_set(client, obj, parsed_args)