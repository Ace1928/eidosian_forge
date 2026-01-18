from itertools import chain
import codecs
from xml.parsers import expat
import six
from six.moves import html_entities as entities, html_parser as html
from genshi.core import Attrs, QName, Stream, stripentities
from genshi.core import START, END, XML_DECL, DOCTYPE, TEXT, START_NS, \
from genshi.compat import StringIO, BytesIO
def XML(text):
    """Parse the given XML source and return a markup stream.
    
    Unlike with `XMLParser`, the returned stream is reusable, meaning it can be
    iterated over multiple times:
    
    >>> xml = XML('<doc><elem>Foo</elem><elem>Bar</elem></doc>')
    >>> print(xml)
    <doc><elem>Foo</elem><elem>Bar</elem></doc>
    >>> print(xml.select('elem'))
    <elem>Foo</elem><elem>Bar</elem>
    >>> print(xml.select('elem/text()'))
    FooBar
    
    :param text: the XML source
    :return: the parsed XML event stream
    :raises ParseError: if the XML text is not well-formed
    """
    return Stream(list(XMLParser(StringIO(text))))