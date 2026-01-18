from neutron_lib._i18n import _
from neutron_lib import exceptions as qexception
class PortDoesNotBelongToTenant(qexception.NotAuthorized):
    message = _('The specified port does not belong to the tenant')