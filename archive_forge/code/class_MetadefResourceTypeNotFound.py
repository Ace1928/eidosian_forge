import urllib.parse as urlparse
from glance.i18n import _
class MetadefResourceTypeNotFound(NotFound):
    message = _('The metadata definition resource-type with name=%(resource_type_name)s, was not found.')