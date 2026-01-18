from oslo_utils import excutils
from neutron_lib._i18n import _
class AddressScopePrefixConflict(Conflict):
    message = _('Failed to associate address scope: subnetpools within an address scope must have unique prefixes.')