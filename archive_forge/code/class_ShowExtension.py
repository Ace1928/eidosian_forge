import logging
from osc_lib.command import command
from osc_lib import utils
from openstackclient.i18n import _
class ShowExtension(command.ShowOne):
    _description = _('Show API extension')

    def get_parser(self, prog_name):
        parser = super(ShowExtension, self).get_parser(prog_name)
        parser.add_argument('extension', metavar='<extension>', help=_('Extension to display. Currently, only network extensions are supported. (Name or Alias)'))
        return parser

    def take_action(self, parsed_args):
        client = self.app.client_manager.network
        extension = client.find_extension(parsed_args.extension, ignore_missing=False)
        display_columns, columns = _get_extension_columns(extension)
        data = utils.get_dict_properties(extension, columns)
        return (display_columns, data)