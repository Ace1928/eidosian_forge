import sys
import xml
import xml.sax.saxutils
from io import StringIO
import docutils
from docutils import frontend, writers, nodes
class TestXml(xml.sax.handler.ContentHandler):

    def setDocumentLocator(self, locator):
        self.locator = locator