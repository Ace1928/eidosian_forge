import sys
from os_win._i18n import _
class ClusterPropertyRetrieveFailed(ClusterException):
    msg_fmt = _('Failed to retrieve a cluster property.')