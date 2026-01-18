from neutron_lib._i18n import _
from neutron_lib import exceptions
class InvalidRoutes(exceptions.InvalidInput):
    message = _('Invalid format for routes: %(routes)s, %(reason)s')