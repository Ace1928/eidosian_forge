import itertools
import logging
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient import exc
from ironicclient.v1 import resource_fields as res_fields
class ShowBaremetalChassis(command.ShowOne):
    """Show chassis details."""
    log = logging.getLogger(__name__ + '.ShowBaremetalChassis')

    def get_parser(self, prog_name):
        parser = super(ShowBaremetalChassis, self).get_parser(prog_name)
        parser.add_argument('chassis', metavar='<chassis>', help=_('UUID of the chassis'))
        parser.add_argument('--fields', nargs='+', dest='fields', metavar='<field>', action='append', choices=res_fields.CHASSIS_DETAILED_RESOURCE.fields, default=[], help=_('One or more chassis fields. Only these fields will be fetched from the server.'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        fields = list(itertools.chain.from_iterable(parsed_args.fields))
        fields = fields if fields else None
        chassis = baremetal_client.chassis.get(parsed_args.chassis, fields=fields)._info
        chassis.pop('links', None)
        chassis.pop('nodes', None)
        return zip(*sorted(chassis.items()))