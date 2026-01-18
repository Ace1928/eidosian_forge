from oslo_utils import excutils
from neutron_lib._i18n import _
class DeviceIDNotOwnedByTenant(Conflict):
    message = _('The following device_id %(device_id)s is not owned by your tenant or matches another tenants router.')