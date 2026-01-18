import argparse
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
class SetPort(common.NeutronCommandWithExtraArgs):
    _description = _('Set port properties')

    def get_parser(self, prog_name):
        parser = super(SetPort, self).get_parser(prog_name)
        _add_updatable_args(parser)
        admin_group = parser.add_mutually_exclusive_group()
        admin_group.add_argument('--enable', action='store_true', default=None, help=_('Enable port'))
        admin_group.add_argument('--disable', action='store_true', help=_('Disable port'))
        parser.add_argument('--name', metavar='<name>', help=_('Set port name'))
        parser.add_argument('--fixed-ip', metavar='subnet=<subnet>,ip-address=<ip-address>', action=parseractions.MultiKeyValueAction, optional_keys=['subnet', 'ip-address'], help=_('Desired IP and/or subnet for this port (name or ID): subnet=<subnet>,ip-address=<ip-address> (repeat option to set multiple fixed IP addresses)'))
        parser.add_argument('--no-fixed-ip', action='store_true', help=_('Clear existing information of fixed IP addresses.Specify both --fixed-ip and --no-fixed-ip to overwrite the current fixed IP addresses.'))
        parser.add_argument('--binding-profile', metavar='<binding-profile>', action=JSONKeyValueAction, help=_('Custom data to be passed as binding:profile. Data may be passed as <key>=<value> or JSON. (repeat option to set multiple binding:profile data)'))
        parser.add_argument('--no-binding-profile', action='store_true', help=_('Clear existing information of binding:profile. Specify both --binding-profile and --no-binding-profile to overwrite the current binding:profile information.'))
        parser.add_argument('--qos-policy', metavar='<qos-policy>', help=_('Attach QoS policy to this port (name or ID)'))
        parser.add_argument('port', metavar='<port>', help=_('Port to modify (name or ID)'))
        parser.add_argument('--security-group', metavar='<security-group>', action='append', dest='security_group', help=_('Security group to associate with this port (name or ID) (repeat option to set multiple security groups)'))
        parser.add_argument('--no-security-group', dest='no_security_group', action='store_true', help=_('Clear existing security groups associated with this port'))
        port_security = parser.add_mutually_exclusive_group()
        port_security.add_argument('--enable-port-security', action='store_true', help=_('Enable port security for this port'))
        port_security.add_argument('--disable-port-security', action='store_true', help=_('Disable port security for this port'))
        parser.add_argument('--allowed-address', metavar='ip-address=<ip-address>[,mac-address=<mac-address>]', action=parseractions.MultiKeyValueAction, dest='allowed_address_pairs', required_keys=['ip-address'], optional_keys=['mac-address'], help=_('Add allowed-address pair associated with this port: ip-address=<ip-address>[,mac-address=<mac-address>] (repeat option to set multiple allowed-address pairs)'))
        parser.add_argument('--no-allowed-address', dest='no_allowed_address_pair', action='store_true', help=_('Clear existing allowed-address pairs associated with this port. (Specify both --allowed-address and --no-allowed-address to overwrite the current allowed-address pairs)'))
        parser.add_argument('--extra-dhcp-option', metavar='name=<name>[,value=<value>,ip-version={4,6}]', default=[], action=parseractions.MultiKeyValueCommaAction, dest='extra_dhcp_options', required_keys=['name'], optional_keys=['value', 'ip-version'], help=_('Extra DHCP options to be assigned to this port: name=<name>[,value=<value>,ip-version={4,6}] (repeat option to set multiple extra DHCP options)'))
        parser.add_argument('--data-plane-status', metavar='<status>', choices=['ACTIVE', 'DOWN'], help=_("Set data plane status of this port (ACTIVE | DOWN). Unset it to None with the 'port unset' command (requires data plane status extension)"))
        _tag.add_tag_option_to_parser_for_set(parser, _('port'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        _prepare_fixed_ips(self.app.client_manager, parsed_args)
        obj = client.find_port(parsed_args.port, ignore_missing=False)
        attrs = _get_attrs(self.app.client_manager, parsed_args)
        if parsed_args.no_binding_profile:
            attrs['binding:profile'] = {}
        if parsed_args.binding_profile:
            if 'binding:profile' not in attrs:
                attrs['binding:profile'] = copy.deepcopy(obj.binding_profile)
            attrs['binding:profile'].update(parsed_args.binding_profile)
        if parsed_args.no_fixed_ip:
            attrs['fixed_ips'] = []
        if parsed_args.fixed_ip:
            if 'fixed_ips' not in attrs:
                attrs['fixed_ips'] = [ip for ip in obj.fixed_ips if ip]
            attrs['fixed_ips'].extend(parsed_args.fixed_ip)
        if parsed_args.no_security_group:
            attrs['security_group_ids'] = []
        if parsed_args.security_group:
            if 'security_group_ids' not in attrs:
                attrs['security_group_ids'] = [id for id in obj.security_group_ids]
            attrs['security_group_ids'].extend((client.find_security_group(sg, ignore_missing=False).id for sg in parsed_args.security_group))
        if parsed_args.no_allowed_address_pair:
            attrs['allowed_address_pairs'] = []
        if parsed_args.allowed_address_pairs:
            if 'allowed_address_pairs' not in attrs:
                attrs['allowed_address_pairs'] = [addr for addr in obj.allowed_address_pairs if addr]
            attrs['allowed_address_pairs'].extend(_convert_address_pairs(parsed_args))
        if parsed_args.extra_dhcp_options:
            attrs['extra_dhcp_opts'] = _convert_extra_dhcp_options(parsed_args)
        if parsed_args.data_plane_status:
            attrs['data_plane_status'] = parsed_args.data_plane_status
        if parsed_args.hint:
            _validate_port_hints(parsed_args.hint)
            expanded_hints = _expand_port_hint_aliases(parsed_args.hint)
            try:
                client.find_extension('port-hints', ignore_missing=False)
            except Exception as e:
                msg = _('Not supported by Network API: %(e)s') % {'e': e}
                raise exceptions.CommandError(msg)
            if 'openvswitch' in expanded_hints and 'other_config' in expanded_hints['openvswitch'] and ('tx-steering' in expanded_hints['openvswitch']['other_config']):
                try:
                    client.find_extension('port-hint-ovs-tx-steering', ignore_missing=False)
                except Exception as e:
                    msg = _('Not supported by Network API: %(e)s') % {'e': e}
                    raise exceptions.CommandError(msg)
            attrs['hints'] = expanded_hints
        attrs.update(self._parse_extra_properties(parsed_args.extra_properties))
        if attrs:
            with common.check_missing_extension_if_error(self.app.client_manager.network, attrs):
                client.update_port(obj, **attrs)
        _tag.update_tags_for_set(client, obj, parsed_args)