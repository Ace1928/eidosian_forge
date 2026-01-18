import json
from osc_lib.command import command
from osc_lib import utils as osc_utils
from oslo_utils import uuidutils
from troveclient import exceptions
from troveclient.i18n import _
class ListDatabaseConfigurationParameters(command.Lister):
    _description = _('Lists available parameters for a configuration group.')
    columns = ['Name', 'Type', 'Min Size', 'Max Size', 'Restart Required']

    def get_parser(self, prog_name):
        parser = super(ListDatabaseConfigurationParameters, self).get_parser(prog_name)
        parser.add_argument('datastore_version', metavar='<datastore_version>', help=_('Datastore version name or ID assigned to the configuration group. ID is preferred if more than one datastore versions have the same name.'))
        parser.add_argument('--datastore', metavar='<datastore>', default=None, help=_('ID or name of the datastore to list configurationparameters for. Optional if the ID of thedatastore_version is provided.'))
        return parser

    def take_action(self, parsed_args):
        db_configuration_parameters = self.app.client_manager.database.configuration_parameters
        if uuidutils.is_uuid_like(parsed_args.datastore_version):
            params = db_configuration_parameters.parameters_by_version(parsed_args.datastore_version)
        elif parsed_args.datastore:
            params = db_configuration_parameters.parameters(parsed_args.datastore, parsed_args.datastore_version)
        else:
            raise exceptions.NoUniqueMatch(_('Either datastore version ID or datastore name needs to be specified.'))
        for param in params:
            setattr(param, 'min_size', getattr(param, 'min', '-'))
            setattr(param, 'max_size', getattr(param, 'max', '-'))
        params = [osc_utils.get_item_properties(p, self.columns) for p in params]
        return (self.columns, params)