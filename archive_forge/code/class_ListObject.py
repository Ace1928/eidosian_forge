import logging
from osc_lib.cli import format_columns
from osc_lib.cli import parseractions
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.common import pagination
from openstackclient.i18n import _
class ListObject(command.Lister):
    _description = _('List objects')

    def get_parser(self, prog_name):
        parser = super(ListObject, self).get_parser(prog_name)
        parser.add_argument('container', metavar='<container>', help=_('Container to list'))
        parser.add_argument('--prefix', metavar='<prefix>', help=_('Filter list using <prefix>'))
        parser.add_argument('--delimiter', metavar='<delimiter>', help=_('Roll up items with <delimiter>'))
        pagination.add_marker_pagination_option_to_parser(parser)
        parser.add_argument('--end-marker', metavar='<end-marker>', help=_('End anchor for paging'))
        parser.add_argument('--long', action='store_true', default=False, help=_('List additional fields in output'))
        parser.add_argument('--all', action='store_true', default=False, help=_('List all objects in container (default is 10000)'))
        return parser

    def take_action(self, parsed_args):
        if parsed_args.long:
            columns = ('Name', 'Bytes', 'Hash', 'Content Type', 'Last Modified')
        else:
            columns = ('Name',)
        kwargs = {}
        if parsed_args.prefix:
            kwargs['prefix'] = parsed_args.prefix
        if parsed_args.delimiter:
            kwargs['delimiter'] = parsed_args.delimiter
        if parsed_args.marker:
            kwargs['marker'] = parsed_args.marker
        if parsed_args.end_marker:
            kwargs['end_marker'] = parsed_args.end_marker
        if parsed_args.limit:
            kwargs['limit'] = parsed_args.limit
        if parsed_args.all:
            kwargs['full_listing'] = True
        data = self.app.client_manager.object_store.object_list(container=parsed_args.container, **kwargs)
        return (columns, (utils.get_dict_properties(s, columns, formatters={}) for s in data))