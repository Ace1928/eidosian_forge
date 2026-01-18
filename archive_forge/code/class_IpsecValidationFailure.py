from neutron_lib._i18n import _
from neutron_lib import exceptions
class IpsecValidationFailure(exceptions.BadRequest):
    message = _("IPSec does not support %(resource)s attribute %(key)s with value '%(value)s'")