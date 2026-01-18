import logging
from osc_lib.cli import format_columns
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ListMetadefNamespace(command.Lister):
    _description = _('List metadef namespaces')

    def get_parser(self, prog_name):
        parser = super().get_parser(prog_name)
        parser.add_argument('--resource-types', metavar='<resource_types>', help=_('filter resource types'))
        parser.add_argument('--visibility', metavar='<visibility>', help=_('filter on visibility'))
        return parser

    def take_action(self, parsed_args):
        image_client = self.app.client_manager.image
        filter_keys = ['resource_types', 'visibility']
        kwargs = {}
        for key in filter_keys:
            argument = getattr(parsed_args, key, None)
            if argument is not None:
                kwargs[key] = argument
        data = list(image_client.metadef_namespaces(**kwargs))
        columns = ['namespace']
        column_headers = columns
        return (column_headers, (utils.get_item_properties(s, columns, formatters=_formatters) for s in data))