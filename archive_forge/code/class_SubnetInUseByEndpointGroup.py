from neutron_lib._i18n import _
from neutron_lib import exceptions
class SubnetInUseByEndpointGroup(exceptions.InUse):
    message = _('Subnet %(subnet_id)s is used by endpoint group %(group_id)s')