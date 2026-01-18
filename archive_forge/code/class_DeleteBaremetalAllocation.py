import itertools
import logging
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.v1 import resource_fields as res_fields
class DeleteBaremetalAllocation(command.Command):
    """Unregister baremetal allocation(s)."""
    log = logging.getLogger(__name__ + '.DeleteBaremetalAllocation')

    def get_parser(self, prog_name):
        parser = super(DeleteBaremetalAllocation, self).get_parser(prog_name)
        parser.add_argument('allocations', metavar='<allocation>', nargs='+', help=_('Allocations(s) to delete (name or UUID).'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        failures = []
        for allocation in parsed_args.allocations:
            try:
                baremetal_client.allocation.delete(allocation)
                print(_('Deleted allocation %s') % allocation)
            except exc.ClientException as e:
                failures.append(_('Failed to delete allocation %(allocation)s:  %(error)s') % {'allocation': allocation, 'error': e})
        if failures:
            raise exc.ClientException('\n'.join(failures))