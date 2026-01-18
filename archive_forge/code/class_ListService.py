import logging
from osc_lib.command import command
from osc_lib import utils
class ListService(command.Lister):
    """Print a list of Magnum services."""
    log = logging.getLogger(__name__ + '.ListService')

    def get_parser(self, prog_name):
        parser = super(ListService, self).get_parser(prog_name)
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        services = client.mservices.list()
        columns = ('id', 'host', 'binary', 'state', 'disabled', 'disabled_reason', 'created_at', 'updated_at')
        return (columns, (utils.get_item_properties(service, columns) for service in services))