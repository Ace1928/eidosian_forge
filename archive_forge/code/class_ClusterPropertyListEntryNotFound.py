import sys
from os_win._i18n import _
class ClusterPropertyListEntryNotFound(ClusterPropertyRetrieveFailed):
    msg_fmt = _("The specified cluster property list does not contain an entry named '%(property_name)s'")