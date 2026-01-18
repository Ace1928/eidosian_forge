from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class UpdateMember(neutronV20.UpdateCommand):
    """LBaaS v2 Update a given member."""
    resource = 'member'
    shadow_resource = 'lbaas_member'

    def add_known_arguments(self, parser):
        parser.add_argument('pool', metavar='POOL', help=_('ID or name of the pool that this member belongs to.'))
        utils.add_boolean_argument(parser, '--admin-state-up', help=_('Update the administrative state of the member (True meaning "Up").'))
        _add_common_args(parser)

    def args2body(self, parsed_args):
        self.parent_id = _get_pool_id(self.get_client(), parsed_args.pool)
        body = {}
        if hasattr(parsed_args, 'admin_state_up'):
            body['admin_state_up'] = parsed_args.admin_state_up
        _parse_common_args(body, parsed_args)
        return {self.resource: body}