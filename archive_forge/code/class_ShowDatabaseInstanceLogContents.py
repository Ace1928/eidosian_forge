from osc_lib.command import command
from osc_lib import utils as osc_utils
from troveclient import exceptions
from troveclient.i18n import _
class ShowDatabaseInstanceLogContents(command.Command):
    _description = _('Show the content of log file.')

    def get_parser(self, prog_name):
        parser = super(ShowDatabaseInstanceLogContents, self).get_parser(prog_name)
        parser.add_argument('instance', metavar='<instance>', type=str, help=_('Id or Name of the instance.'))
        parser.add_argument('log_name', metavar='<log_name>', type=str, help=_('Name of log to operate.'))
        parser.add_argument('--lines', default=50, type=int, help='The number of log lines can be shown in batch.')
        return parser

    def take_action(self, parsed_args):
        db_instances = self.app.client_manager.database.instances
        instance = osc_utils.find_resource(db_instances, parsed_args.instance)
        try:
            log_gen = db_instances.log_generator(instance, parsed_args.log_name, lines=parsed_args.lines)
            for log_part in log_gen():
                print(log_part, end='')
        except exceptions.GuestLogNotFoundError:
            print("ERROR: No published '%(log_name)s' log was found for %(instance)s" % {'log_name': parsed_args.log_name, 'instance': instance})
        except Exception as ex:
            error_msg = ex.message.split('\n')
            print(error_msg[0])