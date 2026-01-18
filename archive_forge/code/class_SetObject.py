import logging
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
class SetObject(command.Command):
    _description = _('Set object properties')

    def get_parser(self, prog_name):
        parser = super(SetObject, self).get_parser(prog_name)
        parser.add_argument('container', metavar='<container>', help=_('Modify <object> from <container>'))
        parser.add_argument('object', metavar='<object>', help=_('Object to modify'))
        parser.add_argument('--property', metavar='<key=value>', required=True, action=parseractions.KeyValueAction, help=_('Set a property on this object (repeat option to set multiple properties)'))
        return parser

    def take_action(self, parsed_args):
        self.app.client_manager.object_store.object_set(parsed_args.container, parsed_args.object, properties=parsed_args.property)