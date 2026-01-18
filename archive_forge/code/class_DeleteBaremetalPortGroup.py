import itertools
import logging
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.v1 import resource_fields as res_fields
class DeleteBaremetalPortGroup(command.Command):
    """Unregister baremetal port group(s)."""
    log = logging.getLogger(__name__ + '.DeleteBaremetalPortGroup')

    def get_parser(self, prog_name):
        parser = super(DeleteBaremetalPortGroup, self).get_parser(prog_name)
        parser.add_argument('portgroups', metavar='<port group>', nargs='+', help=_('Port group(s) to delete (name or UUID).'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        failures = []
        for portgroup in parsed_args.portgroups:
            try:
                baremetal_client.portgroup.delete(portgroup)
                print(_('Deleted port group %s') % portgroup)
            except exc.ClientException as e:
                failures.append(_('Failed to delete port group %(portgroup)s:  %(error)s') % {'portgroup': portgroup, 'error': e})
        if failures:
            raise exc.ClientException('\n'.join(failures))