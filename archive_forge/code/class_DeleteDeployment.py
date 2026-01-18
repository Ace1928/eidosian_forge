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
class DeleteDeployment(command.Command):
    """Delete software deployment(s) and correlative config(s)."""
    log = logging.getLogger(__name__ + '.DeleteDeployment')

    def get_parser(self, prog_name):
        parser = super(DeleteDeployment, self).get_parser(prog_name)
        parser.add_argument('deployment', metavar='<deployment>', nargs='+', help=_('ID of the deployment(s) to delete.'))
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        hc = self.app.client_manager.orchestration
        failure_count = 0
        for deploy_id in parsed_args.deployment:
            try:
                sd = hc.software_deployments.get(deployment_id=deploy_id)
                hc.software_deployments.delete(deployment_id=deploy_id)
            except Exception as e:
                if isinstance(e, heat_exc.HTTPNotFound):
                    print(_('Deployment with ID %s not found') % deploy_id)
                else:
                    print(_('Deployment with ID %s failed to delete') % deploy_id)
                failure_count += 1
                continue
            try:
                config_id = getattr(sd, 'config_id')
                hc.software_configs.delete(config_id=config_id)
            except Exception:
                print(_('Failed to delete the correlative config %(config_id)s of deployment %(deploy_id)s') % {'config_id': config_id, 'deploy_id': deploy_id})
        if failure_count:
            raise exc.CommandError(_('Unable to delete %(count)s of the %(total)s deployments.') % {'count': failure_count, 'total': len(parsed_args.deployment)})