import os
from neutronclient._i18n import _
from neutronclient.neutron import v2_0 as neutronv20
class DeleteQoSPolicy(neutronv20.DeleteCommand):
    """Delete a given qos policy."""
    resource = 'policy'
    shadow_resource = 'qos_policy'