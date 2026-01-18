from oslo_utils import excutils
from neutron_lib._i18n import _
class FailedToAddQdiscToDevice(NeutronException):
    message = _('Failed to add %(direction)s qdisc to device %(device)s.')