import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class SetMetadefObject(command.Command):
    _description = _('Update a metadef object')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('namespace', metavar='<namespace>', help=_('Metadef namespace name'))
        parser.add_argument('object', metavar='<object>', help=_('Metadef object to be updated'))
        parser.add_argument('--name', help=_('New name of the object'))
        return parser

    def take_action(self, parsed_args):
        image_client = self.app.client_manager.image
        object = image_client.get_metadef_object(parsed_args.object, parsed_args.namespace)
        kwargs = {}
        if parsed_args.name:
            kwargs['name'] = parsed_args.name
        image_client.update_metadef_object(object, parsed_args.namespace, **kwargs)