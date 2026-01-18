import os
from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronv20
class ListQoSPolicy(neutronv20.ListCommand):
    """List QoS policies that belong to a given tenant connection."""
    resource = 'policy'
    shadow_resource = 'qos_policy'
    list_columns = ['id', 'name']
    pagination_support = True
    sorting_support = True