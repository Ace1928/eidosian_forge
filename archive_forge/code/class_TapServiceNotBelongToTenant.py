from neutron_lib._i18n import _
from neutron_lib import exceptions as qexception
class TapServiceNotBelongToTenant(qexception.NotAuthorized):
    message = _('Specified Tap Service does not belong to the tenant')