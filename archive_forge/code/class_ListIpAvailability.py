from cliff import show
from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronV20
class ListIpAvailability(neutronV20.ListCommand):
    """List IP usage of networks"""
    resource = 'network_ip_availability'
    resource_plural = 'network_ip_availabilities'
    list_columns = ['network_id', 'network_name', 'total_ips', 'used_ips']
    paginations_support = True
    sorting_support = True
    filter_attrs = [{'name': 'ip_version', 'help': _('Returns IP availability for the network subnets with a given IP version. Default: 4'), 'argparse_kwargs': {'type': int, 'choices': [4, 6], 'default': 4}}, {'name': 'network_id', 'help': _('Returns IP availability for the network matching a given network ID.')}, {'name': 'network_name', 'help': _('Returns IP availability for the network matching a given name.')}, {'name': 'tenant_id', 'help': _('Returns IP availability for the networks with a given tenant ID.')}]