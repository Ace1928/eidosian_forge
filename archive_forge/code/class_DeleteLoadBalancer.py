from oslo_serialization import jsonutils
from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class DeleteLoadBalancer(neutronV20.DeleteCommand):
    """LBaaS v2 Delete a given loadbalancer."""
    resource = 'loadbalancer'