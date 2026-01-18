import urllib.parse as urlparse
from glance.i18n import _
class ProtectedMetadefObjectDelete(Forbidden):
    message = _('Metadata definition object %(object_name)s is protected and cannot be deleted.')