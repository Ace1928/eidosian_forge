import urllib.parse as urlparse
from glance.i18n import _
class MetadefIntegrityError(Forbidden):
    message = _('The metadata definition %(record_type)s with name=%(record_name)s not deleted. Other records still refer to it.')