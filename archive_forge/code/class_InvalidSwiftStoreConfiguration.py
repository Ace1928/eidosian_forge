import urllib.parse as urlparse
from glance.i18n import _
class InvalidSwiftStoreConfiguration(Invalid):
    message = _('Invalid configuration in glance-swift conf file.')