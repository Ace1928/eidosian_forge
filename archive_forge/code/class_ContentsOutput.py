import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class ContentsOutput(ContainerOutput):
    """Outputs the contents converted to HTML"""

    def gethtml(self, container):
        """Return the HTML code"""
        html = []
        if container.contents == None:
            return html
        for element in container.contents:
            if not hasattr(element, 'gethtml'):
                Trace.error('No html in ' + element.__class__.__name__ + ': ' + str(element))
                return html
            html += element.gethtml()
        return html