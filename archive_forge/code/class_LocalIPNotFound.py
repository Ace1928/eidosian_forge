from neutron_lib._i18n import _
from neutron_lib import exceptions
class LocalIPNotFound(exceptions.NotFound):
    message = _('Local IP %(id)s could not be found.')