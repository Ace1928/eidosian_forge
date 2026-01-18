from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import utils
from openstackclient.i18n import _
class ShowTask(command.ShowOne):
    _description = _('Display task details')

    def get_parser(self, prog_name):
        parser = super(ShowTask, self).get_parser(prog_name)
        parser.add_argument('task', metavar='<Task ID>', help=_('Task to display (ID)'))
        return parser

    def take_action(self, parsed_args):
        image_client = self.app.client_manager.image
        task = image_client.get_task(parsed_args.task)
        info = _format_task(task)
        return zip(*sorted(info.items()))