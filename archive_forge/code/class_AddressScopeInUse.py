from neutron_lib._i18n import _
from neutron_lib import exceptions
class AddressScopeInUse(exceptions.InUse):
    message = _('Unable to complete operation on address scope %(address_scope_id)s. There are one or more subnet pools in use on the address scope.')