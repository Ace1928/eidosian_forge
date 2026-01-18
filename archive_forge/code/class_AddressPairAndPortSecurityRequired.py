from neutron_lib._i18n import _
from neutron_lib import exceptions
class AddressPairAndPortSecurityRequired(exceptions.Conflict):
    message = _('Port Security must be enabled in order to have allowed address pairs on a port.')