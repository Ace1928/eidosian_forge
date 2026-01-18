from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from openstackclient.i18n import _
class UnsetAccount(command.Command):
    _description = _('Unset account properties')

    def get_parser(self, prog_name):
        parser = super(UnsetAccount, self).get_parser(prog_name)
        parser.add_argument('--property', metavar='<key>', required=True, action='append', default=[], help=_('Property to remove from account (repeat option to remove multiple properties)'))
        return parser

    def take_action(self, parsed_args):
        self.app.client_manager.object_store.account_unset(properties=parsed_args.property)