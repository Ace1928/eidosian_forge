from neutron_lib._i18n import _
from neutron_lib import exceptions as e
class QosPlacementAllocationConflict(e.Conflict):
    message = _('Allocation for consumer %(consumer)s is not possible on resource provider %(rp)s, the requested amount of bandwidth would exceed the capacity available.')