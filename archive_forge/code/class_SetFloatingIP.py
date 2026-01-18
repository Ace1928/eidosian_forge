from osc_lib import utils
from osc_lib.utils import tags as _tag
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class SetFloatingIP(common.NeutronCommandWithExtraArgs):
    _description = _('Set floating IP Properties')

    def get_parser(self, prog_name):
        parser = super(SetFloatingIP, self).get_parser(prog_name)
        parser.add_argument('floating_ip', metavar='<floating-ip>', help=_('Floating IP to modify (IP address or ID)'))
        (parser.add_argument('--port', metavar='<port>', help=_('Associate the floating IP with port (name or ID)')),)
        parser.add_argument('--fixed-ip-address', metavar='<ip-address>', dest='fixed_ip_address', help=_('Fixed IP of the port (required only if port has multiple IPs)'))
        parser.add_argument('--description', metavar='<description>', help=_('Set floating IP description'))
        qos_policy_group = parser.add_mutually_exclusive_group()
        qos_policy_group.add_argument('--qos-policy', metavar='<qos-policy>', help=_('Attach QoS policy to the floating IP (name or ID)'))
        qos_policy_group.add_argument('--no-qos-policy', action='store_true', help=_('Remove the QoS policy attached to the floating IP'))
        _tag.add_tag_option_to_parser_for_set(parser, _('floating IP'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        attrs = {}
        obj = client.find_ip(parsed_args.floating_ip, ignore_missing=False)
        if parsed_args.port:
            port = client.find_port(parsed_args.port, ignore_missing=False)
            attrs['port_id'] = port.id
        if parsed_args.fixed_ip_address:
            attrs['fixed_ip_address'] = parsed_args.fixed_ip_address
        if parsed_args.description:
            attrs['description'] = parsed_args.description
        if parsed_args.qos_policy:
            attrs['qos_policy_id'] = client.find_qos_policy(parsed_args.qos_policy, ignore_missing=False).id
        if 'no_qos_policy' in parsed_args and parsed_args.no_qos_policy:
            attrs['qos_policy_id'] = None
        attrs.update(self._parse_extra_properties(parsed_args.extra_properties))
        if attrs:
            client.update_ip(obj, **attrs)
        _tag.update_tags_for_set(client, obj, parsed_args)