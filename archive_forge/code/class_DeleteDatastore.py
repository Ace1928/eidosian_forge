from osc_lib.command import command
from osc_lib import utils
from troveclient import exceptions
from troveclient.i18n import _
from troveclient import utils as tc_utils
class DeleteDatastore(command.Command):
    _description = _('Deletes a datastore')

    def get_parser(self, prog_name):
        parser = super(DeleteDatastore, self).get_parser(prog_name)
        parser.add_argument('datastore', metavar='<datastore>', help=_('ID or name of the datastore'))
        return parser

    def take_action(self, parsed_args):
        datastore_client = self.app.client_manager.database.datastores
        try:
            datastore_client.delete(parsed_args.datastore)
        except Exception as e:
            msg = _('Failed to delete datastore %(datastore)s: %(e)s') % {'datastore': parsed_args.datastore, 'e': e}
            raise exceptions.CommandError(msg)