from oslo_utils import excutils
from neutron_lib._i18n import _
class SubnetPoolQuotaExceeded(OverQuota):
    message = _('Per-tenant subnet pool prefix quota exceeded.')