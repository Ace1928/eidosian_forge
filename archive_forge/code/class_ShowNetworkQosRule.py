import itertools
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.network import common
class ShowNetworkQosRule(command.ShowOne):
    _description = _('Display Network QoS rule details')

    def get_parser(self, prog_name):
        parser = super(ShowNetworkQosRule, self).get_parser(prog_name)
        parser.add_argument('qos_policy', metavar='<qos-policy>', help=_('QoS policy that contains the rule (name or ID)'))
        parser.add_argument('id', metavar='<rule-id>', help=_('Network QoS rule to delete (ID)'))
        return parser

    def take_action(self, parsed_args):
        network_client = self.app.client_manager.network
        rule_id = parsed_args.id
        try:
            qos = network_client.find_qos_policy(parsed_args.qos_policy, ignore_missing=False)
            rule_type = _find_rule_type(qos, rule_id)
            if not rule_type:
                raise Exception('Rule not found')
            obj = _rule_action_call(network_client, ACTION_SHOW, rule_type)(rule_id, qos.id)
        except Exception as e:
            msg = _('Failed to set Network QoS rule ID "%(rule)s": %(e)s') % {'rule': rule_id, 'e': e}
            raise exceptions.CommandError(msg)
        display_columns, columns = _get_columns(obj)
        data = utils.get_item_properties(obj, columns)
        return (display_columns, data)