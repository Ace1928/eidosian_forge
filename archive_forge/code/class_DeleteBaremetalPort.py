import itertools
import logging
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.v1 import resource_fields as res_fields
class DeleteBaremetalPort(command.Command):
    """Delete port(s)."""
    log = logging.getLogger(__name__ + '.DeleteBaremetalPort')

    def get_parser(self, prog_name):
        parser = super(DeleteBaremetalPort, self).get_parser(prog_name)
        parser.add_argument('ports', metavar='<port>', nargs='+', help=_('UUID(s) of the port(s) to delete.'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        failures = []
        for port in parsed_args.ports:
            try:
                baremetal_client.port.delete(port)
                print(_('Deleted port %s') % port)
            except exc.ClientException as e:
                failures.append(_('Failed to delete port %(port)s: %(error)s') % {'port': port, 'error': e})
        if failures:
            raise exc.ClientException('\n'.join(failures))