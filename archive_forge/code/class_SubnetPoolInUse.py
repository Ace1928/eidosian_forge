from oslo_utils import excutils
from neutron_lib._i18n import _
class SubnetPoolInUse(InUse):
    """An operational error indicating a subnet pool is still in use.

    A specialization of the InUse exception indicating an operation failed
    on a subnet pool because it's still in use.

    :param subnet_pool_id: The UUID of the subnet pool requested.
    :param reason: Details on why the operation failed. If None a default
        reason is used indicating two or more concurrent subnets are allocated.
    """
    message = _('Unable to complete operation on subnet pool %(subnet_pool_id)s. %(reason)s.')

    def __init__(self, **kwargs):
        if 'reason' not in kwargs:
            kwargs['reason'] = _('Two or more concurrent subnets allocated')
        super().__init__(**kwargs)