import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.network import common
from openstackclient.network import utils as network_utils
class ListDefaultSecurityGroupRule(command.Lister):
    """List security group rules used for new default security groups.

    This shows the rules that will be added to any new default security groups
    created. These rules may differ for the rules present on existing default
    security groups.
    """

    def _format_network_security_group_rule(self, rule):
        """Transform the SDK DefaultSecurityGroupRule object to a dict

        The SDK object gets in the way of reformatting columns...
        Create port_range column from port_range_min and port_range_max
        """
        rule = rule.to_dict()
        rule['port_range'] = network_utils.format_network_port_range(rule)
        rule['remote_ip_prefix'] = network_utils.format_remote_ip_prefix(rule)
        return rule

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('--protocol', metavar='<protocol>', type=network_utils.convert_to_lowercase, help=_('List rules by the IP protocol (ah, dhcp, egp, esp, gre, icmp, igmp, ipv6-encap, ipv6-frag, ipv6-icmp, ipv6-nonxt, ipv6-opts, ipv6-route, ospf, pgm, rsvp, sctp, tcp, udp, udplite, vrrp and integer representations [0-255] or any; default: any (all protocols))'))
        parser.add_argument('--ethertype', metavar='<ethertype>', type=network_utils.convert_to_lowercase, help=_('List default rules by the Ethertype (IPv4 or IPv6)'))
        direction_group = parser.add_mutually_exclusive_group()
        direction_group.add_argument('--ingress', action='store_true', help=_('List default rules which will be applied to incoming network traffic'))
        direction_group.add_argument('--egress', action='store_true', help=_('List default rules which will be applied to outgoing network traffic'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.sdk_connection.network
        column_headers = ('ID', 'IP Protocol', 'Ethertype', 'IP Range', 'Port Range', 'Direction', 'Remote Security Group', 'Remote Address Group', 'Used in default Security Group', 'Used in custom Security Group')
        columns = ('id', 'protocol', 'ether_type', 'remote_ip_prefix', 'port_range', 'direction', 'remote_group_id', 'remote_address_group_id', 'used_in_default_sg', 'used_in_non_default_sg')
        query = {}
        if parsed_args.ingress:
            query['direction'] = 'ingress'
        if parsed_args.egress:
            query['direction'] = 'egress'
        if parsed_args.protocol is not None:
            query['protocol'] = parsed_args.protocol
        rules = [self._format_network_security_group_rule(r) for r in client.default_security_group_rules(**query)]
        return (column_headers, (utils.get_dict_properties(s, columns) for s in rules))