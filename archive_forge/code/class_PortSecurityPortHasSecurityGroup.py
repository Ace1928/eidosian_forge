from neutron_lib._i18n import _
from neutron_lib import exceptions
class PortSecurityPortHasSecurityGroup(exceptions.InUse):
    message = _('Port has security group associated. Cannot disable port security or IP address until security group is removed.')