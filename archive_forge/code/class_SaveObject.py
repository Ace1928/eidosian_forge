import logging
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
class SaveObject(command.Command):
    _description = _('Save object locally')

    def get_parser(self, prog_name):
        parser = super(SaveObject, self).get_parser(prog_name)
        parser.add_argument('--file', metavar='<filename>', help=_("Destination filename (defaults to object name); using '-' as the filename will print the file to stdout"))
        parser.add_argument('container', metavar='<container>', help=_('Download <object> from <container>'))
        parser.add_argument('object', metavar='<object>', help=_('Object to save'))
        return parser

    def take_action(self, parsed_args):
        self.app.client_manager.object_store.object_save(container=parsed_args.container, object=parsed_args.object, file=parsed_args.file)