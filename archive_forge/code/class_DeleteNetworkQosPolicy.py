import logging
from cliff import columns as cliff_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common as identity_common
from openstackclient.network import common
class DeleteNetworkQosPolicy(command.Command):
    _description = _('Delete Qos Policy(s)')

    def get_parser(self, prog_name):
        parser = super(DeleteNetworkQosPolicy, self).get_parser(prog_name)
        parser.add_argument('policy', metavar='<qos-policy>', nargs='+', help=_('QoS policy(s) to delete (name or ID)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        result = 0
        for policy in parsed_args.policy:
            try:
                obj = client.find_qos_policy(policy, ignore_missing=False)
                client.delete_qos_policy(obj)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete QoS policy name or ID '%(qos_policy)s': %(e)s"), {'qos_policy': policy, 'e': e})
        if result > 0:
            total = len(parsed_args.policy)
            msg = _('%(result)s of %(total)s QoS policies failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)