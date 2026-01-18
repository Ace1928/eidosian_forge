import urllib.parse as urlparse
from glance.i18n import _
class ProtectedMetadefResourceTypeSystemDelete(Forbidden):
    message = _('Metadata definition resource-type %(resource_type_name)s is a seeded-system type and cannot be deleted.')