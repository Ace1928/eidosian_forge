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
class UnsetPort(common.NeutronUnsetCommandWithExtraArgs):
    _description = _('Unset port properties')

    def get_parser(self, prog_name):
        parser = super(UnsetPort, self).get_parser(prog_name)
        parser.add_argument('--fixed-ip', metavar='subnet=<subnet>,ip-address=<ip-address>', action=parseractions.MultiKeyValueAction, optional_keys=['subnet', 'ip-address'], help=_('Desired IP and/or subnet which should be removed from this port (name or ID): subnet=<subnet>,ip-address=<ip-address> (repeat option to unset multiple fixed IP addresses)'))
        parser.add_argument('--binding-profile', metavar='<binding-profile-key>', action='append', help=_('Desired key which should be removed from binding:profile (repeat option to unset multiple binding:profile data)'))
        parser.add_argument('--security-group', metavar='<security-group>', action='append', dest='security_group_ids', help=_('Security group which should be removed this port (name or ID) (repeat option to unset multiple security groups)'))
        parser.add_argument('--allowed-address', metavar='ip-address=<ip-address>[,mac-address=<mac-address>]', action=parseractions.MultiKeyValueAction, dest='allowed_address_pairs', required_keys=['ip-address'], optional_keys=['mac-address'], help=_('Desired allowed-address pair which should be removed from this port: ip-address=<ip-address>[,mac-address=<mac-address>] (repeat option to unset multiple allowed-address pairs)'))
        parser.add_argument('--qos-policy', action='store_true', default=False, help=_('Remove the QoS policy attached to the port'))
        parser.add_argument('--data-plane-status', action='store_true', help=_('Clear existing information of data plane status'))
        parser.add_argument('--numa-policy', action='store_true', help=_('Clear existing NUMA affinity policy'))
        parser.add_argument('--host', action='store_true', default=False, help=_('Clear host binding for the port.'))
        parser.add_argument('--hints', action='store_true', default=False, help=_('Clear hints for the port.'))
        _tag.add_tag_option_to_parser_for_unset(parser, _('port'))
        parser.add_argument('port', metavar='<port>', help=_('Port to modify (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        obj = client.find_port(parsed_args.port, ignore_missing=False)
        tmp_fixed_ips = copy.deepcopy(obj.fixed_ips)
        tmp_binding_profile = copy.deepcopy(obj.binding_profile)
        tmp_secgroups = copy.deepcopy(obj.security_group_ids)
        tmp_addr_pairs = copy.deepcopy(obj.allowed_address_pairs)
        _prepare_fixed_ips(self.app.client_manager, parsed_args)
        attrs = {}
        if parsed_args.fixed_ip:
            try:
                for ip in parsed_args.fixed_ip:
                    tmp_fixed_ips.remove(ip)
            except ValueError:
                msg = _('Port does not contain fixed-ip %s') % ip
                raise exceptions.CommandError(msg)
            attrs['fixed_ips'] = tmp_fixed_ips
        if parsed_args.binding_profile:
            try:
                for key in parsed_args.binding_profile:
                    del tmp_binding_profile[key]
            except KeyError:
                msg = _('Port does not contain binding-profile %s') % key
                raise exceptions.CommandError(msg)
            attrs['binding:profile'] = tmp_binding_profile
        if parsed_args.security_group_ids:
            try:
                for sg in parsed_args.security_group_ids:
                    sg_id = client.find_security_group(sg, ignore_missing=False).id
                    tmp_secgroups.remove(sg_id)
            except ValueError:
                msg = _('Port does not contain security group %s') % sg
                raise exceptions.CommandError(msg)
            attrs['security_group_ids'] = tmp_secgroups
        if parsed_args.allowed_address_pairs:
            try:
                for addr in _convert_address_pairs(parsed_args):
                    tmp_addr_pairs.remove(addr)
            except ValueError:
                msg = _('Port does not contain allowed-address-pair %s') % addr
                raise exceptions.CommandError(msg)
            attrs['allowed_address_pairs'] = tmp_addr_pairs
        if parsed_args.qos_policy:
            attrs['qos_policy_id'] = None
        if parsed_args.data_plane_status:
            attrs['data_plane_status'] = None
        if parsed_args.numa_policy:
            attrs['numa_affinity_policy'] = None
        if parsed_args.host:
            attrs['binding:host_id'] = None
        if parsed_args.hints:
            attrs['hints'] = None
        attrs.update(self._parse_extra_properties(parsed_args.extra_properties))
        if attrs:
            client.update_port(obj, **attrs)
        _tag.update_tags_for_unset(client, obj, parsed_args)