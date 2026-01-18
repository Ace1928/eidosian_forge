from oslo_log import log as logging
from osc_lib.command import command
from osc_lib import utils
class ListAvailabilityZone(command.Lister):
    """List availability zones"""
    log = logging.getLogger(__name__ + '.ListAvailabilityZones')

    def get_parser(self, prog_name):
        parser = super(ListAvailabilityZone, self).get_parser(prog_name)
        return parser

    def take_action(self, parsed_args):
        client = _get_client(self, parsed_args)
        zones = client.availability_zones.list()
        columns = ('availability_zone',)
        return (columns, (utils.get_item_properties(zone, columns) for zone in zones))