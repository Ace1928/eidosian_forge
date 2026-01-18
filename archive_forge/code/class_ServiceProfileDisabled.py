from neutron_lib._i18n import _
from neutron_lib import exceptions
class ServiceProfileDisabled(exceptions.ServiceUnavailable):
    message = _('Service Profile is not enabled.')