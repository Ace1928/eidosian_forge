import urllib.parse as urlparse
from glance.i18n import _
class MetadefDuplicateResourceType(Duplicate):
    message = _('A metadata definition resource-type with name=%(resource_type_name)s already exists.')