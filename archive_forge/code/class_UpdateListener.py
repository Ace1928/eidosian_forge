from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class UpdateListener(neutronV20.UpdateCommand):
    """LBaaS v2 Update a given listener."""
    resource = 'listener'

    def add_known_arguments(self, parser):
        _add_common_args(parser)
        parser.add_argument('--name', help=_('Name of the listener.'))
        utils.add_boolean_argument(parser, '--admin-state-up', dest='admin_state_up', help=_('Specify the administrative state of the listener. (True meaning "Up")'))

    def args2body(self, parsed_args):
        body = {}
        neutronV20.update_dict(parsed_args, body, ['admin_state_up'])
        _parse_common_args(body, parsed_args, self.get_client())
        return {self.resource: body}