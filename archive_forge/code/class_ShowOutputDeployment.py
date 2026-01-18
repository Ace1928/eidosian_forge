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
class ShowOutputDeployment(command.Command):
    """Show a specific deployment output."""
    log = logging.getLogger(__name__ + '.ShowOutputDeployment')

    def get_parser(self, prog_name):
        parser = super(ShowOutputDeployment, self).get_parser(prog_name)
        parser.add_argument('deployment', metavar='<deployment>', help=_('ID of deployment to show the output for'))
        parser.add_argument('output', metavar='<output-name>', nargs='?', default=None, help=_('Name of an output to display'))
        parser.add_argument('--all', default=False, action='store_true', help=_('Display all deployment outputs'))
        parser.add_argument('--long', action='store_true', default=False, help='Show full deployment logs in output')
        return parser

    def take_action(self, parsed_args):
        self.log.debug('take_action(%s)', parsed_args)
        heat_client = self.app.client_manager.orchestration
        if not parsed_args.all and parsed_args.output is None or (parsed_args.all and parsed_args.output is not None):
            raise exc.CommandError(_('Error: either %(output)s or %(all)s argument is needed.') % {'output': '<output-name>', 'all': '--all'})
        try:
            sd = heat_client.software_deployments.get(deployment_id=parsed_args.deployment)
        except heat_exc.HTTPNotFound:
            raise exc.CommandError(_('Deployment not found: %s') % parsed_args.deployment)
        outputs = sd.output_values
        if outputs:
            if parsed_args.all:
                print('output_values:\n')
                for k in outputs:
                    format_utils.print_software_deployment_output(data=outputs, name=k, long=parsed_args.long)
            elif parsed_args.output not in outputs:
                msg = _('Output %(output)s does not exist in deployment %(deployment)s') % {'output': parsed_args.output, 'deployment': parsed_args.deployment}
                raise exc.CommandError(msg)
            else:
                print('output_value:\n')
                format_utils.print_software_deployment_output(data=outputs, name=parsed_args.output)