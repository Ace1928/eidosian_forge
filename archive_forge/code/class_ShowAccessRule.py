import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
from openstackclient.identity import common
class ShowAccessRule(command.ShowOne):
    _description = _('Display access rule details')

    def get_parser(self, prog_name):
        parser = super(ShowAccessRule, self).get_parser(prog_name)
        parser.add_argument('access_rule', metavar='<access-rule>', help=_('Access rule ID to display'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        access_rule = common.get_resource_by_id(identity_client.access_rules, parsed_args.access_rule)
        access_rule._info.pop('links', None)
        return zip(*sorted(access_rule._info.items()))