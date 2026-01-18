import logging
from cliff import columns as cliff_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class SetNetworkQosPolicy(common.NeutronCommandWithExtraArgs):
    _description = _('Set QoS policy properties')

    def get_parser(self, prog_name):
        parser = super(SetNetworkQosPolicy, self).get_parser(prog_name)
        parser.add_argument('policy', metavar='<qos-policy>', help=_('QoS policy to modify (name or ID)'))
        parser.add_argument('--name', metavar='<name>', help=_('Set QoS policy name'))
        parser.add_argument('--description', metavar='<description>', help=_('Description of the QoS policy'))
        enable_group = parser.add_mutually_exclusive_group()
        enable_group.add_argument('--share', action='store_true', help=_('Make the QoS policy accessible by other projects'))
        enable_group.add_argument('--no-share', action='store_true', help=_('Make the QoS policy not accessible by other projects'))
        default_group = parser.add_mutually_exclusive_group()
        default_group.add_argument('--default', action='store_true', help=_('Set this as a default network QoS policy'))
        default_group.add_argument('--no-default', action='store_true', help=_('Set this as a non-default network QoS policy'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        obj = client.find_qos_policy(parsed_args.policy, ignore_missing=False)
        attrs = _get_attrs(self.app.client_manager, parsed_args)
        attrs.update(self._parse_extra_properties(parsed_args.extra_properties))
        client.update_qos_policy(obj, **attrs)