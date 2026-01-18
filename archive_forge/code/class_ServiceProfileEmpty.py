from neutron_lib._i18n import _
from neutron_lib import exceptions
class ServiceProfileEmpty(exceptions.InvalidInput):
    message = _('Service Profile needs either a driver or metainfo.')