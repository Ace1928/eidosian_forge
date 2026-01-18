import logging
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class DeleteMetadefNamespace(command.Command):
    _description = _('Delete metadef namespace')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('namespace', metavar='<namespace>', nargs='+', help=_('Metadef namespace(s) to delete (name)'))
        return parser

    def take_action(self, parsed_args):
        image_client = self.app.client_manager.image
        result = 0
        for ns in parsed_args.namespace:
            try:
                namespace = image_client.get_metadef_namespace(ns)
                image_client.delete_metadef_namespace(namespace.id)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete namespace with name or ID '%(namespace)s': %(e)s"), {'namespace': ns, 'e': e})
        if result > 0:
            total = len(parsed_args.namespace)
            msg = _('%(result)s of %(total)s namespace failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)