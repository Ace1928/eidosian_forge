import urllib.parse as urlparse
from glance.i18n import _
class MetadefTagNotFound(NotFound):
    message = _('The metadata definition tag with name=%(name)s was not found in namespace=%(namespace_name)s.')