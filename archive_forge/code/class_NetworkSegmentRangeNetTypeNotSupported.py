from neutron_lib._i18n import _
from neutron_lib import exceptions
class NetworkSegmentRangeNetTypeNotSupported(exceptions.BadRequest):
    message = _('Network type %(type)s does not support network segment ranges.')