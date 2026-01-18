import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class CreateMetadefObjects(command.ShowOne):
    _description = _('Create a metadef object')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('--namespace', metavar='<namespace>', help=_('Metadef namespace to create the object in (name)'))
        parser.add_argument('name', metavar='<metadef-object-name>', help=_('New metadef object name'))
        return parser

    def take_action(self, parsed_args):
        image_client = self.app.client_manager.image
        namespace = image_client.get_metadef_namespace(parsed_args.namespace)
        data = image_client.create_metadef_object(namespace=namespace.namespace, name=parsed_args.name)
        fields, value = _format_object(data)
        return (fields, value)