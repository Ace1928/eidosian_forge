import itertools
import logging
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient.v1 import resource_fields as res_fields
from ironicclient.v1 import utils as v1_utils
class ListBaremetalDriverRaidProperty(command.Lister):
    """List a driver's RAID logical disk properties."""
    log = logging.getLogger(__name__ + '.ListBaremetalDriverRaidProperty')

    def get_parser(self, prog_name):
        parser = super(ListBaremetalDriverRaidProperty, self).get_parser(prog_name)
        parser.add_argument('driver', metavar='<driver>', help='Name of the driver.')
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        raid_props = baremetal_client.driver.raid_logical_disk_properties(parsed_args.driver)
        labels = ['Property', 'Description']
        return (labels, sorted(raid_props.items()))