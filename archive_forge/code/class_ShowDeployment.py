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
class ShowDeployment(command.ShowOne):
    """Show SoftwareDeployment Details."""
    log = logging.getLogger(__name__ + '.ShowSoftwareDeployment')

    def get_parser(self, prog_name):
        parser = super(ShowDeployment, self).get_parser(prog_name)
        parser.add_argument('deployment', metavar='<deployment>', help=_('ID of the deployment'))
        parser.add_argument('--long', action='store_true', help=_('Show more fields in output'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        heat_client = self.app.client_manager.orchestration
        try:
            data = heat_client.software_deployments.get(deployment_id=parsed_args.deployment)
        except heat_exc.HTTPNotFound:
            raise exc.CommandError(_('Software Deployment not found: %s') % parsed_args.deployment)
        else:
            columns = ['id', 'server_id', 'config_id', 'creation_time', 'updated_time', 'status', 'status_reason', 'input_values', 'action']
            if parsed_args.long:
                columns.append('output_values')
            return (columns, utils.get_item_properties(data, columns))