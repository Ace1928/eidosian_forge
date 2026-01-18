from neutron_lib._i18n import _
from neutron_lib import exceptions
class HANetworkConcurrentDeletion(exceptions.Conflict):
    message = _('Network for tenant %(tenant_id)s concurrently deleted.')