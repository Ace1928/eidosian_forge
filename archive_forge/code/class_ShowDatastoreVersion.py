from osc_lib.command import command
from osc_lib import utils
from troveclient import exceptions
from troveclient.i18n import _
from troveclient import utils as tc_utils
class ShowDatastoreVersion(command.ShowOne):
    _description = _('Shows details of a datastore version.')

    def get_parser(self, prog_name):
        parser = super(ShowDatastoreVersion, self).get_parser(prog_name)
        parser.add_argument('datastore_version', metavar='<datastore_version>', help=_('ID or name of the datastore version.'))
        parser.add_argument('--datastore', metavar='<datastore>', default=None, help=_('ID or name of the datastore. Optional if the ID ofthe datastore_version is provided.'))
        return parser

    def take_action(self, parsed_args):
        datastore_version_client = self.app.client_manager.database.datastore_versions
        if parsed_args.datastore:
            datastore_version = datastore_version_client.get(parsed_args.datastore, parsed_args.datastore_version)
        elif tc_utils.is_uuid_like(parsed_args.datastore_version):
            datastore_version = datastore_version_client.get_by_uuid(parsed_args.datastore_version)
        else:
            raise exceptions.NoUniqueMatch(_('The datastore name or id is required to retrieve a datastore version by name.'))
        if datastore_version._info.get('links'):
            del datastore_version._info['links']
        return zip(*sorted(datastore_version._info.items()))