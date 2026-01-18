from oslo_utils import excutils
from neutron_lib._i18n import _
class HostRoutesExhausted(BadRequest):
    message = _('Unable to complete operation for %(subnet_id)s. The number of host routes exceeds the limit %(quota)s.')