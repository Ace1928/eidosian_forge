import urllib.parse as urlparse
from glance.i18n import _
class ProtectedMetadefResourceTypeAssociationDelete(Forbidden):
    message = _('Metadata definition resource-type-association %(resource_type)s is protected and cannot be deleted.')