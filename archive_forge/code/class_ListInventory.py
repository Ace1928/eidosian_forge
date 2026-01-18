import collections
import itertools
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib.i18n import _
from osc_lib import utils
from oslo_utils import excutils
from osc_placement.resources import common
from osc_placement import version
class ListInventory(command.Lister):
    """List inventories for a given resource provider."""

    def get_parser(self, prog_name):
        parser = super(ListInventory, self).get_parser(prog_name)
        parser.add_argument('uuid', metavar='<uuid>', help='UUID of the resource provider')
        return parser

    def take_action(self, parsed_args):
        http = self.app.client_manager.placement
        url = BASE_URL.format(uuid=parsed_args.uuid)
        resources = http.request('GET', url).json()
        inventories = [dict(resource_class=k, **v) for k, v in resources['inventories'].items()]
        url = USAGES_BASE_URL.format(uuid=parsed_args.uuid)
        resources = http.request('GET', url).json()['usages']
        for inventory in inventories:
            inventory['used'] = resources[inventory['resource_class']]
        fields = ('resource_class',) + FIELDS + ('used',)
        rows = (utils.get_dict_properties(i, fields) for i in inventories)
        return (fields, rows)