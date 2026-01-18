import json
from osc_lib.command import command
from osc_lib import utils as osc_utils
from oslo_utils import uuidutils
from troveclient import exceptions
from troveclient.i18n import _
class UpdateDatabaseConfiguration(command.Command):
    _description = _('Update a configuration group.')

    def get_parser(self, prog_name):
        parser = super(UpdateDatabaseConfiguration, self).get_parser(prog_name)
        parser.add_argument('configuration_group_id', help=_('Configuration group ID.'))
        parser.add_argument('values', metavar='<values>', help=_('Dictionary of the values to set.'))
        parser.add_argument('--name', metavar='<name>', help=_('New name of the configuration group.'))
        parser.add_argument('--description', metavar='<description>', default=None, help=_('An optional description for the configuration group.'))
        return parser

    def take_action(self, parsed_args):
        db_configurations = self.app.client_manager.database.configurations
        db_configurations.update(parsed_args.configuration_group_id, parsed_args.values, name=parsed_args.name, description=parsed_args.description)