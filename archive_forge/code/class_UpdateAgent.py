from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronV20
class UpdateAgent(neutronV20.UpdateCommand):
    """Updates the admin status and description for a specified agent."""
    resource = 'agent'
    allow_names = False

    def add_known_arguments(self, parser):
        parser.add_argument('--admin-state-down', dest='admin_state', action='store_false', help=_('Set admin state up of the agent to false.'))
        parser.add_argument('--description', help=_('Description for the agent.'))

    def args2body(self, parsed_args):
        body = {'admin_state_up': parsed_args.admin_state}
        neutronV20.update_dict(parsed_args, body, ['description'])
        return {self.resource: body}