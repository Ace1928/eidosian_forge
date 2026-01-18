import itertools
import logging
from osc_lib.command import command
from osc_lib import utils as oscutils
from ironicclient.common.i18n import _
from ironicclient.common import utils
from ironicclient.v1 import resource_fields as res_fields
from ironicclient.v1 import utils as v1_utils
class PassthruListBaremetalDriver(command.Lister):
    """List available vendor passthru methods for a driver."""
    log = logging.getLogger(__name__ + '.PassthruListBaremetalDriver')

    def get_parser(self, prog_name):
        parser = super(PassthruListBaremetalDriver, self).get_parser(prog_name)
        parser.add_argument('driver', metavar='<driver>', help=_('Name of the driver.'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        baremetal_client = self.app.client_manager.baremetal
        columns = res_fields.VENDOR_PASSTHRU_METHOD_RESOURCE.fields
        labels = res_fields.VENDOR_PASSTHRU_METHOD_RESOURCE.labels
        methods = baremetal_client.driver.get_vendor_passthru_methods(parsed_args.driver)
        params = []
        for method, response in methods.items():
            response['name'] = method
            http_methods = ', '.join(response['http_methods'])
            response['http_methods'] = http_methods
            params.append(response)
        return (labels, (oscutils.get_dict_properties(s, columns) for s in params))