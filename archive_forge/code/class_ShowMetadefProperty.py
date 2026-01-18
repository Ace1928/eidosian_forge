import json
import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ShowMetadefProperty(command.ShowOne):
    _description = _('Show a particular metadef property')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('namespace', metavar='<namespace>', help=_('Metadef namespace of the property (name)'))
        parser.add_argument('property', metavar='<property>', help=_('Property to show'))
        return parser

    def take_action(self, parsed_args):
        image_client = self.app.client_manager.image
        data = image_client.get_metadef_property(parsed_args.property, parsed_args.namespace)
        info = _format_property(data)
        return zip(*sorted(info.items()))