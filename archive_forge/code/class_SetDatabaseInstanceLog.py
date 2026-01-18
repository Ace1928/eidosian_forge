from osc_lib.command import command
from osc_lib import utils as osc_utils
from troveclient import exceptions
from troveclient.i18n import _
class SetDatabaseInstanceLog(command.ShowOne):
    _description = _('Instructs Trove guest to operate logs.')

    def get_parser(self, prog_name):
        parser = super(SetDatabaseInstanceLog, self).get_parser(prog_name)
        parser.add_argument('instance', metavar='<instance>', type=str, help=_('Id or Name of the instance.'))
        parser.add_argument('log_name', metavar='<log_name>', type=str, help=_('Name of log to operate.'))
        parser.add_argument('--enable', action='store_true', help='Whether or not to enable log collection.')
        parser.add_argument('--disable', action='store_true', help='Whether or not to disable log collection.')
        parser.add_argument('--publish', action='store_true', help='Whether or not to publish log files to the backend storage for logs(Swift by default).')
        parser.add_argument('--discard', action='store_true', help='Whether or not to discard the existing logs before publish.')
        return parser

    def take_action(self, parsed_args):
        db_instances = self.app.client_manager.database.instances
        instance = osc_utils.find_resource(db_instances, parsed_args.instance)
        log_info = db_instances.log_action(instance, parsed_args.log_name, enable=parsed_args.enable, disable=parsed_args.disable, discard=parsed_args.discard, publish=parsed_args.publish)
        result = log_info._info
        return zip(*sorted(result.items()))