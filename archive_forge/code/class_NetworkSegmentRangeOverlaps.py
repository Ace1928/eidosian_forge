from neutron_lib._i18n import _
from neutron_lib import exceptions
class NetworkSegmentRangeOverlaps(exceptions.Conflict):
    message = _('Network segment range overlaps with range(s) with id %(range_id)s')