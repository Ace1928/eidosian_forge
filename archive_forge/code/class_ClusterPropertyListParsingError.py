import sys
from os_win._i18n import _
class ClusterPropertyListParsingError(ClusterPropertyRetrieveFailed):
    msg_fmt = _('Parsing a cluster property list failed.')