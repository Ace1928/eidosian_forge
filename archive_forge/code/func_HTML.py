from itertools import chain
import codecs
from xml.parsers import expat
import six
from six.moves import html_entities as entities, html_parser as html
from genshi.core import Attrs, QName, Stream, stripentities
from genshi.core import START, END, XML_DECL, DOCTYPE, TEXT, START_NS, \
from genshi.compat import StringIO, BytesIO
def HTML(text, encoding=None):
    """Parse the given HTML source and return a markup stream.
    
    Unlike with `HTMLParser`, the returned stream is reusable, meaning it can be
    iterated over multiple times:
    
    >>> html = HTML('<body><h1>Foo</h1></body>', encoding='utf-8')
    >>> print(html)
    <body><h1>Foo</h1></body>
    >>> print(html.select('h1'))
    <h1>Foo</h1>
    >>> print(html.select('h1/text()'))
    Foo
    
    :param text: the HTML source
    :return: the parsed XML event stream
    :raises ParseError: if the HTML text is not well-formed, and error recovery
                        fails
    """
    if isinstance(text, six.text_type):
        return Stream(list(HTMLParser(StringIO(text), encoding=None)))
    return Stream(list(HTMLParser(BytesIO(text), encoding=encoding)))