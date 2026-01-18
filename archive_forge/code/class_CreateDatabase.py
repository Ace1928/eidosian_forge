from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from troveclient.i18n import _
class CreateDatabase(command.Command):
    _description = _('Creates a database on an instance.')

    def get_parser(self, prog_name):
        parser = super(CreateDatabase, self).get_parser(prog_name)
        parser.add_argument('instance', metavar='<instance>', help=_('ID or name of the instance.'))
        parser.add_argument('name', metavar='<name>', help=_('Name of the database.'))
        parser.add_argument('--character_set', metavar='<character_set>', help=_('Optional character set for database.'))
        parser.add_argument('--collate', metavar='<collate>', help=_('Optional collation type for database.'))
        return parser

    def take_action(self, parsed_args):
        manager = self.app.client_manager.database
        databases = manager.databases
        instance = utils.find_resource(manager.instances, parsed_args.instance)
        database_dict = {'name': parsed_args.name}
        if parsed_args.collate:
            database_dict['collate'] = parsed_args.collate
        if parsed_args.character_set:
            database_dict['character_set'] = parsed_args.character_set
        databases.create(instance, [database_dict])