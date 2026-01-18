from neutron_lib._i18n import _
from neutron_lib import exceptions
class AddressGroupInUse(exceptions.InUse):
    message = _('Address group %(address_group_id)s is in use on one or more security group rules.')