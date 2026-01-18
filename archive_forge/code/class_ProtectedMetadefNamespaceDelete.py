import urllib.parse as urlparse
from glance.i18n import _
class ProtectedMetadefNamespaceDelete(Forbidden):
    message = _('Metadata definition namespace %(namespace)s is protected and cannot be deleted.')