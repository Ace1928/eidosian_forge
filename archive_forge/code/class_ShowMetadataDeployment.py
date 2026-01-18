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
class ShowMetadataDeployment(command.Command):
    """Get deployment configuration metadata for the specified server."""
    log = logging.getLogger(__name__ + '.ShowMetadataDeployment')

    def get_parser(self, prog_name):
        parser = super(ShowMetadataDeployment, self).get_parser(prog_name)
        parser.add_argument('server', metavar='<server>', help=_('ID of the server to fetch deployments for'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        heat_client = self.app.client_manager.orchestration
        md = heat_client.software_deployments.metadata(server_id=parsed_args.server)
        print(jsonutils.dumps(md, indent=2))