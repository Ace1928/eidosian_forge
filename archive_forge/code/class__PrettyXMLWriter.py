from __future__ import unicode_literals
import io
from xml.sax.saxutils import XMLGenerator
from xml.sax.xmlreader import AttributesImpl
import six
from pybtex.database.output import BaseWriter
class _PrettyXMLWriter(object):

    def __init__(self, output, encoding='UTF-8', namespace=('bibtex', 'http://bibtexml.sf.net/'), header=True):
        self.prefix, self.uri = namespace
        self.generator = XMLGenerator(output, encoding=encoding)
        if header:
            self.generator.startDocument()
        self.generator.startPrefixMapping(self.prefix, self.uri)
        self.stack = []

    def write(self, data):
        self.generator.characters(data)

    def newline(self):
        self.write('\n')

    def indent_line(self):
        self.write(' ' * (len(self.stack) * 4))

    def start(self, tag, attrs=None, newline=True):
        if attrs is None:
            attrs = {}
        else:
            attrs = {(None, key): value for key, value in attrs.items()}
        self.indent_line()
        self.stack.append(tag)
        self.generator.startElementNS((self.uri, tag), tag, AttributesImpl(attrs))
        if newline:
            self.newline()

    def end(self, indent=True):
        tag = self.stack.pop()
        if indent:
            self.indent_line()
        self.generator.endElementNS((self.uri, tag), tag)
        self.newline()

    def element(self, tag, data):
        self.start(tag, newline=False)
        self.write(data)
        self.end(indent=False)

    def close(self):
        self.generator.endDocument()