from neutron_lib._i18n import _
from neutron_lib import exceptions
class CsrValidationFailure(exceptions.BadRequest):
    message = _("Cisco CSR does not support %(resource)s attribute %(key)s with value '%(value)s'")