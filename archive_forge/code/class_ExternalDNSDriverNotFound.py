from neutron_lib._i18n import _
from neutron_lib import exceptions
class ExternalDNSDriverNotFound(exceptions.NotFound):
    message = _('External DNS driver %(driver)s could not be found.')