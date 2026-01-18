import itertools
import logging
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient import exc
from ironicclient.v1 import resource_fields as res_fields
class ShowBaremetalConductor(command.ShowOne):
    """Show baremetal conductor details"""
    log = logging.getLogger(__name__ + '.ShowBaremetalConductor')

    def get_parser(self, prog_name):
        parser = super(ShowBaremetalConductor, self).get_parser(prog_name)
        parser.add_argument('conductor', metavar='<conductor>', help=_('Hostname of the conductor'))
        parser.add_argument('--fields', nargs='+', dest='fields', metavar='<field>', action='append', choices=res_fields.CONDUCTOR_DETAILED_RESOURCE.fields, default=[], help=_('One or more conductor fields. Only these fields will be fetched from the server.'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        fields = list(itertools.chain.from_iterable(parsed_args.fields))
        fields = fields if fields else None
        conductor = baremetal_client.conductor.get(parsed_args.conductor, fields=fields)._info
        conductor.pop('links', None)
        return self.dict2columns(conductor)