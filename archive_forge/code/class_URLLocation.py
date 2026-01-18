import os
import sys
import zipfile
import weakref
from io import BytesIO
import pyglet
class URLLocation(Location):
    """Location on the network.

    This class uses the ``urlparse`` and ``urllib2`` modules to open files on
    the network given a URL.
    """

    def __init__(self, base_url):
        """Create a location given a base URL.

        :Parameters:
            `base_url` : str
                URL string to prepend to filenames.

        """
        self.base = base_url

    def open(self, filename, mode='rb'):
        import urllib.parse
        import urllib.request
        url = urllib.parse.urljoin(self.base, filename)
        return urllib.request.urlopen(url)