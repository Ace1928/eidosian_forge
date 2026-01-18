import urllib.parse as urlparse
from glance.i18n import _
class RedirectException(Exception):

    def __init__(self, url):
        self.url = urlparse.urlparse(url)