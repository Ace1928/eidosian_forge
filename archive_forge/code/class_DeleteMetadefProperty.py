import json
import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class DeleteMetadefProperty(command.Command):
    _description = _('Delete metadef propert(ies)')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('namespace', metavar='<namespace>', help=_('Metadef namespace of the property (name)'))
        parser.add_argument('properties', metavar='<property>', nargs='+', help=_('Metadef propert(ies) to delete (name)'))
        return parser

    def take_action(self, parsed_args):
        image_client = self.app.client_manager.image
        result = 0
        for prop in parsed_args.properties:
            try:
                image_client.delete_metadef_property(prop, parsed_args.namespace, ignore_missing=False)
            except Exception as e:
                result += 1
                LOG.error(_("Failed to delete property with name or ID '%(property)s' from namespace '%(namespace)s': %(e)s"), {'property': prop, 'namespace': parsed_args.namespace, 'e': e})
        if result > 0:
            total = len(parsed_args.namespace)
            msg = _('%(result)s of %(total)s properties failed to delete.') % {'result': result, 'total': total}
            raise exceptions.CommandError(msg)