from neutron_lib._i18n import _
from neutron_lib import exceptions
class DVRMacAddressNotFound(exceptions.NotFound):
    message = _('Distributed Virtual Router Mac Address for host %(host)s does not exist.')