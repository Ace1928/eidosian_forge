import argparse
from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import exceptions
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.neutron.v2_0 import dns
from neutronclient.neutron.v2_0.qos import policy as qos_policy
class ListPort(neutronV20.ListCommand):
    """List ports that belong to a given tenant."""
    resource = 'port'
    _formatters = {'fixed_ips': _format_fixed_ips}
    list_columns = ['id', 'name', 'mac_address', 'fixed_ips']
    pagination_support = True
    sorting_support = True