import itertools
import logging
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.v1 import resource_fields as res_fields
class ShowBaremetalAllocation(command.ShowOne):
    """Show baremetal allocation details."""
    log = logging.getLogger(__name__ + '.ShowBaremetalAllocation')

    def get_parser(self, prog_name):
        parser = super(ShowBaremetalAllocation, self).get_parser(prog_name)
        parser.add_argument('allocation', metavar='<id>', help=_('UUID or name of the allocation'))
        parser.add_argument('--fields', nargs='+', dest='fields', metavar='<field>', action='append', choices=res_fields.ALLOCATION_DETAILED_RESOURCE.fields, default=[], help=_('One or more allocation fields. Only these fields will be fetched from the server.'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        fields = list(itertools.chain.from_iterable(parsed_args.fields))
        fields = fields if fields else None
        allocation = baremetal_client.allocation.get(parsed_args.allocation, fields=fields)._info
        allocation.pop('links', None)
        return zip(*sorted(allocation.items()))