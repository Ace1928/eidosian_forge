from oslo_utils import excutils
from neutron_lib._i18n import _
class OverQuota(Conflict):
    """A error due to exceeding quota limits.

    A specialization of the Conflict exception indicating quota has been
    exceeded.

    :param overs: The resources that have exceeded quota.
    """
    message = _('Quota exceeded for resources: %(overs)s.')