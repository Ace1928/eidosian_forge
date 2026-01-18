import json
import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class CreateMetadefProperty(command.ShowOne):
    _description = _('Create a metadef property')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('--name', required=True, help=_('Internal name of the property'))
        parser.add_argument('--title', required=True, help=_('Property name displayed to the user'))
        parser.add_argument('--type', required=True, help=_('Property type'))
        parser.add_argument('--schema', required=True, help=_('Valid JSON schema of the property'))
        parser.add_argument('namespace', help=_('Name of namespace the property will belong.'))
        return parser

    def take_action(self, parsed_args):
        image_client = self.app.client_manager.image
        kwargs = {'name': parsed_args.name, 'title': parsed_args.title, 'type': parsed_args.type}
        try:
            kwargs.update(json.loads(parsed_args.schema))
        except json.JSONDecodeError as e:
            raise exceptions.CommandError(_('Failed to load JSON schema: %(e)s') % {'e': e})
        data = image_client.create_metadef_property(parsed_args.namespace, **kwargs)
        info = _format_property(data)
        return zip(*sorted(info.items()))