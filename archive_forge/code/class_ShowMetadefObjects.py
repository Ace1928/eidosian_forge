import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ShowMetadefObjects(command.ShowOne):
    _description = _('Show a particular metadef object')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('namespace', metavar='<namespace>', help=_('Metadef namespace of the object (name)'))
        parser.add_argument('object', metavar='<object>', help=_('Metadef object to show'))
        return parser

    def take_action(self, parsed_args):
        image_client = self.app.client_manager.image
        namespace = parsed_args.namespace
        object = parsed_args.object
        data = image_client.get_metadef_object(object, namespace)
        fields, value = _format_object(data)
        return (fields, value)