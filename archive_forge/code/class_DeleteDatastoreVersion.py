from osc_lib.command import command
from osc_lib import utils
from troveclient import exceptions
from troveclient.i18n import _
from troveclient import utils as tc_utils
class DeleteDatastoreVersion(command.Command):
    _description = _('Deletes a datastore version.')

    def get_parser(self, prog_name):
        parser = super(DeleteDatastoreVersion, self).get_parser(prog_name)
        parser.add_argument('datastore_version', metavar='<datastore_version>', help=_('ID of the datastore version.'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.database.mgmt_ds_versions
        try:
            client.delete(parsed_args.datastore_version)
        except Exception as e:
            msg = _('Failed to delete datastore version %(version)s: %(e)s') % {'version': parsed_args.datastore_version, 'e': e}
            raise exceptions.CommandError(msg)