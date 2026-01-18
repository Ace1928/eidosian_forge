import itertools
import logging
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient.v1 import resource_fields as res_fields
from ironicclient.v1 import utils as v1_utils
class ListBaremetalDriver(command.Lister):
    """List the enabled drivers."""
    log = logging.getLogger(__name__ + '.ListBaremetalDriver')

    def get_parser(self, prog_name):
        parser = super(ListBaremetalDriver, self).get_parser(prog_name)
        parser.add_argument('--type', metavar='<type>', choices=['classic', 'dynamic'], help='Type of driver ("classic" or "dynamic"). The default is to list all of them.')
        display_group = parser.add_mutually_exclusive_group()
        display_group.add_argument('--long', action='store_true', default=None, help='Show detailed information about the drivers.')
        display_group.add_argument('--fields', nargs='+', dest='fields', metavar='<field>', action='append', default=[], choices=res_fields.DRIVER_DETAILED_RESOURCE.fields, help=_("One or more node fields. Only these fields will be fetched from the server. Can not be used when '--long' is specified."))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        client = self.app.client_manager.baremetal
        params = {'driver_type': parsed_args.type, 'detail': parsed_args.long}
        if parsed_args.long:
            labels = res_fields.DRIVER_DETAILED_RESOURCE.labels
            columns = res_fields.DRIVER_DETAILED_RESOURCE.fields
        elif parsed_args.fields:
            fields = itertools.chain.from_iterable(parsed_args.fields)
            resource = res_fields.Resource(list(fields))
            columns = resource.fields
            labels = resource.labels
            params['fields'] = columns
        else:
            labels = res_fields.DRIVER_RESOURCE.labels
            columns = res_fields.DRIVER_RESOURCE.fields
        drivers = client.driver.list(**params)
        drivers = oscutils.sort_items(drivers, 'name')
        data = [utils.convert_list_props_to_comma_separated(d._info) for d in drivers]
        return (labels, (oscutils.get_dict_properties(s, columns) for s in data))