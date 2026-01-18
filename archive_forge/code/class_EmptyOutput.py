import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class EmptyOutput(ContainerOutput):

    def gethtml(self, container):
        """Return empty HTML code."""
        return []

    def isempty(self):
        """This output is particularly empty."""
        return True