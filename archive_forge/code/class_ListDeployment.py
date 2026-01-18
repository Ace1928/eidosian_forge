import logging
from osc_lib.command import command
from osc_lib import exceptions as exc
from osc_lib import utils
from oslo_serialization import jsonutils
from heatclient._i18n import _
from heatclient.common import deployment_utils
from heatclient.common import format_utils
from heatclient.common import utils as heat_utils
from heatclient import exc as heat_exc
class ListDeployment(command.Lister):
    """List software deployments."""
    log = logging.getLogger(__name__ + '.ListDeployment')

    def get_parser(self, prog_name):
        parser = super(ListDeployment, self).get_parser(prog_name)
        parser.add_argument('--server', metavar='<server>', help=_('ID of the server to fetch deployments for'))
        parser.add_argument('--long', action='store_true', help=_('List more fields in output'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        heat_client = self.app.client_manager.orchestration
        return _list_deployment(heat_client, args=parsed_args)