from oslo_utils import excutils
from neutron_lib._i18n import _
class PortBoundNUMAAffinityPolicy(InUse):
    """An operational error indicating a port is already bound.

    NUMA affinity policy cannot be modified when the port is bound.

    :param port_id: The UUID of the port requested.
    :param host_id: The host ID where the port is bound.
    :param numa_affinity_policy: value passed to be updated.
    """
    message = _('Unable to complete operation on port %(port_id)s, port is already bound to host %(host_id)s, numa_affinity_policy value given %(numa_affinity_policy)s.')