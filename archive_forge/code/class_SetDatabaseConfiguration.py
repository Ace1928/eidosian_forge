import json
from osc_lib.command import command
from osc_lib import utils as osc_utils
from oslo_utils import uuidutils
from troveclient import exceptions
from troveclient.i18n import _
class SetDatabaseConfiguration(command.Command):
    _description = _('Change parameters for a configuration group.')

    def get_parser(self, prog_name):
        parser = super(SetDatabaseConfiguration, self).get_parser(prog_name)
        parser.add_argument('configuration_group_id', help=_('Configuration group ID.'))
        parser.add_argument('values', metavar='<values>', help=_('Dictionary of the new values to set.'))
        return parser

    def take_action(self, parsed_args):
        db_configurations = self.app.client_manager.database.configurations
        db_configurations.edit(parsed_args.configuration_group_id, parsed_args.values)