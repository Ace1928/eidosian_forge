from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronv20
class CreateEndpointGroup(neutronv20.CreateCommand):
    """Create a VPN endpoint group."""
    resource = 'endpoint_group'

    def add_known_arguments(self, parser):
        add_known_endpoint_group_arguments(parser)

    def subnet_name2id(self, endpoint):
        return neutronv20.find_resourceid_by_name_or_id(self.get_client(), 'subnet', endpoint)

    def args2body(self, parsed_args):
        if parsed_args.type == 'subnet':
            endpoints = [self.subnet_name2id(ep) for ep in parsed_args.endpoints]
        else:
            endpoints = parsed_args.endpoints
        body = {'endpoints': endpoints}
        neutronv20.update_dict(parsed_args, body, ['name', 'description', 'tenant_id', 'type'])
        return {self.resource: body}