from neutron_lib._i18n import _
from neutron_lib import exceptions
class CsrInternalError(exceptions.NeutronException):
    message = _('Fatal - %(reason)s')