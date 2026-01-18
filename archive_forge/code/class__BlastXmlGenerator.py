import re
import warnings
from itertools import chain
from xml.etree import ElementTree
from xml.sax.saxutils import XMLGenerator, escape
from Bio import BiopythonParserWarning
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
class _BlastXmlGenerator(XMLGenerator):
    """Event-based XML Generator."""

    def __init__(self, out, encoding='utf-8', indent=' ', increment=2):
        """Initialize the class."""
        XMLGenerator.__init__(self, out, encoding)
        self._indent = indent
        self._level = 0
        self._increment = increment
        self._parent_stack = []

    def startDocument(self):
        """Start the XML document."""
        self._write('<?xml version="1.0"?>\n<!DOCTYPE BlastOutput PUBLIC "-//NCBI//NCBI BlastOutput/EN" "http://www.ncbi.nlm.nih.gov/dtd/NCBI_BlastOutput.dtd">\n')

    def startElement(self, name, attrs=None, children=False):
        """Start an XML element.

        :param name: element name
        :type name: string
        :param attrs: element attributes
        :type attrs: dictionary {string: object}
        :param children: whether the element has children or not
        :type children: bool

        """
        if attrs is None:
            attrs = {}
        self.ignorableWhitespace(self._indent * self._level)
        XMLGenerator.startElement(self, name, attrs)

    def endElement(self, name):
        """End and XML element of the given name."""
        XMLGenerator.endElement(self, name)
        self._write('\n')

    def startParent(self, name, attrs=None):
        """Start an XML element which has children.

        :param name: element name
        :type name: string
        :param attrs: element attributes
        :type attrs: dictionary {string: object}

        """
        if attrs is None:
            attrs = {}
        self.startElement(name, attrs, children=True)
        self._level += self._increment
        self._write('\n')
        self._parent_stack.append(name)

    def endParent(self):
        """End an XML element with children."""
        name = self._parent_stack.pop()
        self._level -= self._increment
        self.ignorableWhitespace(self._indent * self._level)
        self.endElement(name)

    def startParents(self, *names):
        """Start XML elements without children."""
        for name in names:
            self.startParent(name)

    def endParents(self, num):
        """End XML elements, according to the given number."""
        for i in range(num):
            self.endParent()

    def simpleElement(self, name, content=None):
        """Create an XML element without children with the given content."""
        self.startElement(name, attrs={})
        if content:
            self.characters(content)
        self.endElement(name)

    def characters(self, content):
        """Replace quotes and apostrophe."""
        content = escape(str(content))
        for a, b in (('"', '&quot;'), ("'", '&apos;')):
            content = content.replace(a, b)
        self._write(content)