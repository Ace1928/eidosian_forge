import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.i18n import _
class ListRegion(command.Lister):
    _description = _('List regions')

    def get_parser(self, prog_name):
        parser = super(ListRegion, self).get_parser(prog_name)
        parser.add_argument('--parent-region', metavar='<region-id>', help=_('Filter by parent region ID'))
        return parser

    def take_action(self, parsed_args):
        identity_client = self.app.client_manager.identity
        kwargs = {}
        if parsed_args.parent_region:
            kwargs['parent_region_id'] = parsed_args.parent_region
        columns_headers = ('Region', 'Parent Region', 'Description')
        columns = ('ID', 'Parent Region Id', 'Description')
        data = identity_client.regions.list(**kwargs)
        return (columns_headers, (utils.get_item_properties(s, columns, formatters={}) for s in data))