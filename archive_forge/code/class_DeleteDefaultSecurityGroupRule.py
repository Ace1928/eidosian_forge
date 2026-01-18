import logging
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.network import common
from openstackclient.network import utils as network_utils
class DeleteDefaultSecurityGroupRule(command.Command):
    """Remove security group rule(s) from the default security group template.

    These rules will not longer be applied to the default security groups
    created for any new project. They will not be removed from any existing
    default security groups.
    """

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('rule', metavar='<rule>', nargs='+', help=_('Default security group rule(s) to delete (ID only)'))
        return parser

    def take_action(self, parsed_args):
        result = 0
        client = self.app.client_manager.sdk_connection.network
        for r in parsed_args.rule:
            try:
                obj = client.find_default_security_group_rule(r, ignore_missing=False)
                client.delete_default_security_group_rule(obj)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete default SG rule with ID '%(rule_id)s': %(e)s"), {'rule_id': r, 'e': e})
        if result > 0:
            total = len(parsed_args.rule)
            msg = _('%(result)s of %(total)s default rules failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)