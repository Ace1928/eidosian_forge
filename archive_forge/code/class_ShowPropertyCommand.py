import ast
import logging
from cliff import command
from cliff.formatters import table
from cliff import lister
from cliff import show
from blazarclient import exception
from blazarclient import utils
class ShowPropertyCommand(BlazarCommand, show.ShowOne):
    """Show information of a given resource property."""
    api = 'reservation'
    resource = None
    log = None

    def get_parser(self, prog_name):
        parser = super(ShowPropertyCommand, self).get_parser(prog_name)
        parser.add_argument('property_name', metavar='PROPERTY_NAME', help='Name of property.')
        return parser

    def get_data(self, parsed_args):
        self.log.debug('get_data(%s)' % parsed_args)
        blazar_client = self.get_client()
        resource_manager = getattr(blazar_client, self.resource)
        data = resource_manager.get_property(parsed_args.property_name)
        if parsed_args.formatter == 'table':
            self.format_output_data(data)
        return list(zip(*sorted(data.items())))