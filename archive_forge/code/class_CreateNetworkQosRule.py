import itertools
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.network import common
class CreateNetworkQosRule(command.ShowOne, common.NeutronCommandWithExtraArgs):
    _description = _('Create new Network QoS rule')

    def get_parser(self, prog_name):
        parser = super(CreateNetworkQosRule, self).get_parser(prog_name)
        parser.add_argument('qos_policy', metavar='<qos-policy>', help=_('QoS policy that contains the rule (name or ID)'))
        parser.add_argument('--type', metavar='<type>', required=True, choices=[RULE_TYPE_MINIMUM_BANDWIDTH, RULE_TYPE_MINIMUM_PACKET_RATE, RULE_TYPE_DSCP_MARKING, RULE_TYPE_BANDWIDTH_LIMIT], help=_('QoS rule type (%s)') % ', '.join(MANDATORY_PARAMETERS.keys()))
        _add_rule_arguments(parser)
        return parser

    def take_action(self, parsed_args):
        network_client = self.app.client_manager.network
        try:
            attrs = _get_attrs(network_client, parsed_args, is_create=True)
            attrs.update(self._parse_extra_properties(parsed_args.extra_properties))
            obj = _rule_action_call(network_client, ACTION_CREATE, parsed_args.type)(attrs.pop('qos_policy_id'), **attrs)
        except Exception as e:
            msg = _('Failed to create Network QoS rule: %(e)s') % {'e': e}
            raise exceptions.CommandError(msg)
        display_columns, columns = _get_columns(obj)
        data = utils.get_item_properties(obj, columns)
        return (display_columns, data)