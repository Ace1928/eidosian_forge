from oslo_utils import excutils
from neutron_lib._i18n import _
class OverlappingAllocationPools(Conflict):
    message = _('Found overlapping allocation pools: %(pool_1)s %(pool_2)s for subnet %(subnet_cidr)s.')