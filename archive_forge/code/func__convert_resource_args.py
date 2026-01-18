from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.neutron import v2_0 as neutronv20
def _convert_resource_args(client, parsed_args):
    resource_type = client.get_resource_plural(parsed_args.resource_type)
    resource_id = neutronv20.find_resourceid_by_name_or_id(client, parsed_args.resource_type, parsed_args.resource)
    return (resource_type, resource_id)