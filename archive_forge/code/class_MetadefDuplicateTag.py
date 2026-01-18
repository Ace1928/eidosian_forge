import urllib.parse as urlparse
from glance.i18n import _
class MetadefDuplicateTag(Duplicate):
    message = _('A metadata tag with name=%(name)s already exists in namespace=%(namespace_name)s. (Please note that metadata tag names are case insensitive).')