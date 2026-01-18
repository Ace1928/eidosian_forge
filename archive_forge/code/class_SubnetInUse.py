from oslo_utils import excutils
from neutron_lib._i18n import _
class SubnetInUse(InUse):
    """An operational error indicating a subnet is still in use.

    A specialization of the InUse exception indicating an operation failed
    on a subnet because the subnet is still in use.

    :param subnet_id: The UUID of the subnet requested.
    :param reason: Details on why the operation failed. If None, a default
        reason is used indicating one or more ports still have IP allocations
        on the subnet.
    """
    message = _('Unable to complete operation on subnet %(subnet_id)s: %(reason)s.')

    def __init__(self, **kwargs):
        if 'reason' not in kwargs:
            kwargs['reason'] = _('One or more ports have an IP allocation from this subnet')
        super().__init__(**kwargs)