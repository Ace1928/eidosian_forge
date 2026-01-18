import urllib.parse as urlparse
from glance.i18n import _
class FailedToGetScrubberJobs(GlanceException):
    message = _('Scrubber encountered an error while trying to fetch scrub jobs.')