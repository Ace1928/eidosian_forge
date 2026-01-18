from neutron_lib._i18n import _
from neutron_lib import exceptions
class NetworkSegmentRangeReferencedByProject(exceptions.InUse):
    message = _('Network Segment Range %(range_id)s is referenced by one or more tenant networks.')