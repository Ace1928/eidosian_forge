import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class DeleteAccessRule(command.Command):
    _description = _('Delete access rule(s)')

    def get_parser(self, prog_name):
        parser = super(DeleteAccessRule, self).get_parser(prog_name)
        parser.add_argument('access_rule', metavar='<access-rule>', nargs='+', help=_('Access rule ID(s) to delete'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        errors = 0
        for ac in parsed_args.access_rule:
            try:
                access_rule = common.get_resource_by_id(identity_client.access_rules, ac)
                identity_client.access_rules.delete(access_rule.id)
            except Exception as e:
                errors += 1
                LOG.error(_("Failed to delete access rule with ID '%(ac)s': %(e)s"), {'ac': ac, 'e': e})
        if errors > 0:
            total = len(parsed_args.access_rule)
            msg = _('%(errors)s of %(total)s access rules failed to delete.') % {'errors': errors, 'total': total}
            raise exceptions.CommandError(msg)