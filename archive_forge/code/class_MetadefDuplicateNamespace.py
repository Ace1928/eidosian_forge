import urllib.parse as urlparse
from glance.i18n import _
class MetadefDuplicateNamespace(Duplicate):
    message = _('The metadata definition namespace=%(namespace_name)s already exists.')