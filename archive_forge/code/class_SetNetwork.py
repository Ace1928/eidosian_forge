from cliff import columns as cliff_columns
from osc_lib.cli import format_columns
from osc_lib import utils
from osc_lib.utils import tags as _tag
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class SetNetwork(common.NeutronCommandWithExtraArgs):
    _description = _('Set network properties')

    def get_parser(self, prog_name):
        parser = super(SetNetwork, self).get_parser(prog_name)
        parser.add_argument('network', metavar='<network>', help=_('Network to modify (name or ID)'))
        parser.add_argument('--name', metavar='<name>', help=_('Set network name'))
        admin_group = parser.add_mutually_exclusive_group()
        admin_group.add_argument('--enable', action='store_true', default=None, help=_('Enable network'))
        admin_group.add_argument('--disable', action='store_true', help=_('Disable network'))
        share_group = parser.add_mutually_exclusive_group()
        share_group.add_argument('--share', action='store_true', default=None, help=_('Share the network between projects'))
        share_group.add_argument('--no-share', action='store_true', help=_('Do not share the network between projects'))
        parser.add_argument('--description', metavar='<description>', help=_('Set network description'))
        parser.add_argument('--mtu', metavar='<mtu>', help=_('Set network mtu'))
        port_security_group = parser.add_mutually_exclusive_group()
        port_security_group.add_argument('--enable-port-security', action='store_true', help=_('Enable port security by default for ports created on this network'))
        port_security_group.add_argument('--disable-port-security', action='store_true', help=_('Disable port security by default for ports created on this network'))
        external_router_grp = parser.add_mutually_exclusive_group()
        external_router_grp.add_argument('--external', action='store_true', help=_("The network has an external routing facility that's not managed by Neutron and can be used as in: openstack router set --external-gateway NETWORK (external-net extension required)"))
        external_router_grp.add_argument('--internal', action='store_true', help=_("Opposite of '--external'"))
        default_router_grp = parser.add_mutually_exclusive_group()
        default_router_grp.add_argument('--default', action='store_true', help=_('Set the network as the default external network'))
        default_router_grp.add_argument('--no-default', action='store_true', help=_('Do not use the network as the default external network'))
        qos_group = parser.add_mutually_exclusive_group()
        qos_group.add_argument('--qos-policy', metavar='<qos-policy>', help=_('QoS policy to attach to this network (name or ID)'))
        qos_group.add_argument('--no-qos-policy', action='store_true', help=_('Remove the QoS policy attached to this network'))
        _tag.add_tag_option_to_parser_for_set(parser, _('network'))
        _add_additional_network_options(parser)
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        obj = client.find_network(parsed_args.network, ignore_missing=False)
        attrs = _get_attrs_network(self.app.client_manager, parsed_args)
        attrs.update(self._parse_extra_properties(parsed_args.extra_properties))
        if attrs:
            with common.check_missing_extension_if_error(self.app.client_manager.network, attrs):
                client.update_network(obj, **attrs)
        _tag.update_tags_for_set(client, obj, parsed_args)