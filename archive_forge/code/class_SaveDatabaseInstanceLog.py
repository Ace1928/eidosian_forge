from osc_lib.command import command
from osc_lib import utils as osc_utils
from troveclient import exceptions
from troveclient.i18n import _
class SaveDatabaseInstanceLog(command.Command):
    _description = _('Save the log file.')

    def get_parser(self, prog_name):
        parser = super(SaveDatabaseInstanceLog, self).get_parser(prog_name)
        parser.add_argument('instance', metavar='<instance>', type=str, help=_('Id or Name of the instance.'))
        parser.add_argument('log_name', metavar='<log_name>', type=str, help=_('Name of log to operate.'))
        parser.add_argument('--file', help='Path of file to save log to for instance.')
        return parser

    def take_action(self, parsed_args):
        db_instances = self.app.client_manager.database.instances
        instance = osc_utils.find_resource(db_instances, parsed_args.instance)
        try:
            filepath = db_instances.log_save(instance, parsed_args.log_name, filename=parsed_args.file)
            print(_('Log "%(log_name)s" written to %(file_name)s') % {'log_name': parsed_args.log_name, 'file_name': filepath})
        except exceptions.GuestLogNotFoundError:
            print("ERROR: No published '%(log_name)s' log was found for %(instance)s" % {'log_name': parsed_args.log_name, 'instance': instance})
        except Exception as ex:
            error_msg = ex.message.split('\n')
            print(error_msg[0])