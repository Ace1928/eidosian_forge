from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import utils
from openstackclient.i18n import _
class ListTask(command.Lister):
    _description = _('List tasks')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('--sort-key', metavar='<key>[:<field>]', help=_('Sorts the response by one of the following attributes: created_at, expires_at, id, status, type, updated_at. (default is created_at) (multiple keys and directions can be specified separated by comma)'))
        parser.add_argument('--sort-dir', metavar='<key>[:<direction>]', help=_('Sort output by selected keys and directions (asc or desc) (default: name:desc) (multiple keys and directions can be specified separated by comma)'))
        parser.add_argument('--limit', metavar='<num-tasks>', type=int, help=_('Maximum number of tasks to display.'))
        parser.add_argument('--marker', metavar='<task>', help=_('The last task of the previous page. Display list of tasks after marker. Display all tasks if not specified. (name or ID)'))
        parser.add_argument('--type', metavar='<type>', choices=['import'], help=_('Filters the response by a task type.'))
        parser.add_argument('--status', metavar='<status>', choices=['pending', 'processing', 'success', 'failure'], help=_('Filter tasks based on status.'))
        return parser

    def take_action(self, parsed_args):
        image_client = self.app.client_manager.image
        columns = ('id', 'type', 'status', 'owner_id')
        column_headers = ('ID', 'Type', 'Status', 'Owner')
        kwargs = {}
        copy_attrs = {'sort_key', 'sort_dir', 'limit', 'marker', 'type', 'status'}
        for attr in copy_attrs:
            val = getattr(parsed_args, attr, None)
            if val is not None:
                kwargs[attr] = val
        data = image_client.tasks(**kwargs)
        return (column_headers, (utils.get_item_properties(s, columns, formatters=_formatters) for s in data))