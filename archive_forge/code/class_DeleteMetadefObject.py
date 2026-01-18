import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class DeleteMetadefObject(command.Command):
    _description = _('Delete metadata definitions object(s)')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('namespace', metavar='<namespace>', help=_('Metadef namespace of the object (name)'))
        parser.add_argument('objects', metavar='<object>', nargs='+', help=_('Metadef object(s) to delete (name)'))
        return parser

    def take_action(self, parsed_args):
        image_client = self.app.client_manager.image
        namespace = parsed_args.namespace
        result = 0
        for obj in parsed_args.objects:
            try:
                object = image_client.get_metadef_object(obj, namespace)
                image_client.delete_metadef_object(object, namespace)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete object with name or ID '%(object)s': %(e)s"), {'object': obj, 'e': e})
        if result > 0:
            total = len(parsed_args.namespace)
            msg = _('%(result)s of %(total)s object failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)