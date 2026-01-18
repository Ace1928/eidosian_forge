from neutron_lib._i18n import _
from neutron_lib import exceptions
class AddressesAlreadyExist(exceptions.BadRequest):
    message = _('Addresses %(addresses)s already exist in the address group %(address_group_id)s.')