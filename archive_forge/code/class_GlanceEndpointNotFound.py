import urllib.parse as urlparse
from glance.i18n import _
class GlanceEndpointNotFound(NotFound):
    message = _('%(interface)s glance endpoint not found for region %(region)s')