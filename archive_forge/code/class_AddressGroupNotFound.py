from neutron_lib._i18n import _
from neutron_lib import exceptions
class AddressGroupNotFound(exceptions.NotFound):
    message = _('Address group %(address_group_id)s could not be found.')