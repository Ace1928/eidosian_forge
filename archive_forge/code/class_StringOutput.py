import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class StringOutput(ContainerOutput):
    """Returns a bare string as output"""

    def gethtml(self, container):
        """Return a bare string"""
        return [container.string]