import itertools
import logging
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.v1 import resource_fields as res_fields
class DeleteBaremetalChassis(command.Command):
    """Delete a chassis."""
    log = logging.getLogger(__name__ + '.DeleteBaremetalChassis')

    def get_parser(self, prog_name):
        parser = super(DeleteBaremetalChassis, self).get_parser(prog_name)
        parser.add_argument('chassis', metavar='<chassis>', nargs='+', help=_('UUIDs of chassis to delete'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        failures = []
        for chassis in parsed_args.chassis:
            try:
                baremetal_client.chassis.delete(chassis)
                print(_('Deleted chassis %s') % chassis)
            except exc.ClientException as e:
                failures.append(_('Failed to delete chassis %(chassis)s: %(error)s') % {'chassis': chassis, 'error': e})
        if failures:
            raise exc.ClientException('\n'.join(failures))