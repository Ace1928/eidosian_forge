from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import utils
from openstackclient.i18n import _
class ShowCapability(command.Lister):
    _description = _('Show capability command')

    def get_parser(self, prog_name):
        parser = super(ShowCapability, self).get_parser(prog_name)
        parser.add_argument('host', metavar='<host>', help=_('List capabilities of specified host (host@backend-name)'))
        return parser

    def take_action(self, parsed_args):
        volume_client = self.app.client_manager.sdk_connection.volume
        columns = ['Title', 'Key', 'Type', 'Description']
        data = volume_client.get_capabilities(parsed_args.host)
        print_data = []
        keys = data.properties
        for key in keys:
            capability_data = data.properties[key]
            capability_data['key'] = key
            print_data.append(capability_data)
        return (columns, (utils.get_dict_properties(s, columns) for s in print_data))