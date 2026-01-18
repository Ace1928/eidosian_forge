import json
import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ListMetadefProperties(command.Lister):
    _description = _('List metadef properties')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('namespace', metavar='<namespace>', help=_('An identifier (a name) for the namespace'))
        return parser

    def take_action(self, parsed_args):
        image_client = self.app.client_manager.image
        props = image_client.metadef_properties(parsed_args.namespace)
        columns = ['name', 'title', 'type']
        return (columns, (utils.get_item_properties(prop, columns) for prop in props))