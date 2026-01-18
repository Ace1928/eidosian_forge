from functools import reduce
import sys
from itertools import chain
import operator
import six
from genshi.compat import stringrepr
from genshi.util import stripentities, striptags
class Stream(object):
    """Represents a stream of markup events.
    
    This class is basically an iterator over the events.
    
    Stream events are tuples of the form::
    
      (kind, data, position)
    
    where ``kind`` is the event kind (such as `START`, `END`, `TEXT`, etc),
    ``data`` depends on the kind of event, and ``position`` is a
    ``(filename, line, offset)`` tuple that contains the location of the
    original element or text in the input. If the original location is unknown,
    ``position`` is ``(None, -1, -1)``.
    
    Also provided are ways to serialize the stream to text. The `serialize()`
    method will return an iterator over generated strings, while `render()`
    returns the complete generated text at once. Both accept various parameters
    that impact the way the stream is serialized.
    """
    __slots__ = ['events', 'serializer']
    START = StreamEventKind('START')
    END = StreamEventKind('END')
    TEXT = StreamEventKind('TEXT')
    XML_DECL = StreamEventKind('XML_DECL')
    DOCTYPE = StreamEventKind('DOCTYPE')
    START_NS = StreamEventKind('START_NS')
    END_NS = StreamEventKind('END_NS')
    START_CDATA = StreamEventKind('START_CDATA')
    END_CDATA = StreamEventKind('END_CDATA')
    PI = StreamEventKind('PI')
    COMMENT = StreamEventKind('COMMENT')

    def __init__(self, events, serializer=None):
        """Initialize the stream with a sequence of markup events.
        
        :param events: a sequence or iterable providing the events
        :param serializer: the default serialization method to use for this
                           stream

        :note: Changed in 0.5: added the `serializer` argument
        """
        self.events = events
        self.serializer = serializer

    def __iter__(self):
        return iter(self.events)

    def __or__(self, function):
        """Override the "bitwise or" operator to apply filters or serializers
        to the stream, providing a syntax similar to pipes on Unix shells.
        
        Assume the following stream produced by the `HTML` function:
        
        >>> from genshi.input import HTML
        >>> html = HTML('''<p onclick="alert('Whoa')">Hello, world!</p>''', encoding='utf-8')
        >>> print(html)
        <p onclick="alert('Whoa')">Hello, world!</p>
        
        A filter such as the HTML sanitizer can be applied to that stream using
        the pipe notation as follows:
        
        >>> from genshi.filters import HTMLSanitizer
        >>> sanitizer = HTMLSanitizer()
        >>> print(html | sanitizer)
        <p>Hello, world!</p>
        
        Filters can be any function that accepts and produces a stream (where
        a stream is anything that iterates over events):
        
        >>> def uppercase(stream):
        ...     for kind, data, pos in stream:
        ...         if kind is TEXT:
        ...             data = data.upper()
        ...         yield kind, data, pos
        >>> print(html | sanitizer | uppercase)
        <p>HELLO, WORLD!</p>
        
        Serializers can also be used with this notation:
        
        >>> from genshi.output import TextSerializer
        >>> output = TextSerializer()
        >>> print(html | sanitizer | uppercase | output)
        HELLO, WORLD!
        
        Commonly, serializers should be used at the end of the "pipeline";
        using them somewhere in the middle may produce unexpected results.
        
        :param function: the callable object that should be applied as a filter
        :return: the filtered stream
        :rtype: `Stream`
        """
        return Stream(_ensure(function(self)), serializer=self.serializer)

    def filter(self, *filters):
        """Apply filters to the stream.
        
        This method returns a new stream with the given filters applied. The
        filters must be callables that accept the stream object as parameter,
        and return the filtered stream.
        
        The call::
        
            stream.filter(filter1, filter2)
        
        is equivalent to::
        
            stream | filter1 | filter2
        
        :param filters: one or more callable objects that should be applied as
                        filters
        :return: the filtered stream
        :rtype: `Stream`
        """
        return reduce(operator.or_, (self,) + filters)

    def render(self, method=None, encoding=None, out=None, **kwargs):
        """Return a string representation of the stream.
        
        Any additional keyword arguments are passed to the serializer, and thus
        depend on the `method` parameter value.
        
        :param method: determines how the stream is serialized; can be either
                       "xml", "xhtml", "html", "text", or a custom serializer
                       class; if `None`, the default serialization method of
                       the stream is used
        :param encoding: how the output string should be encoded; if set to
                         `None`, this method returns a `unicode` object
        :param out: a file-like object that the output should be written to
                    instead of being returned as one big string; note that if
                    this is a file or socket (or similar), the `encoding` must
                    not be `None` (that is, the output must be encoded)
        :return: a `str` or `unicode` object (depending on the `encoding`
                 parameter), or `None` if the `out` parameter is provided
        :rtype: `basestring`
        
        :see: XMLSerializer, XHTMLSerializer, HTMLSerializer, TextSerializer
        :note: Changed in 0.5: added the `out` parameter
        """
        from genshi.output import encode
        if method is None:
            method = self.serializer or 'xml'
        generator = self.serialize(method=method, **kwargs)
        return encode(generator, method=method, encoding=encoding, out=out)

    def select(self, path, namespaces=None, variables=None):
        """Return a new stream that contains the events matching the given
        XPath expression.
        
        >>> from genshi import HTML
        >>> stream = HTML('<doc><elem>foo</elem><elem>bar</elem></doc>', encoding='utf-8')
        >>> print(stream.select('elem'))
        <elem>foo</elem><elem>bar</elem>
        >>> print(stream.select('elem/text()'))
        foobar
        
        Note that the outermost element of the stream becomes the *context
        node* for the XPath test. That means that the expression "doc" would
        not match anything in the example above, because it only tests against
        child elements of the outermost element:
        
        >>> print(stream.select('doc'))
        <BLANKLINE>
        
        You can use the "." expression to match the context node itself
        (although that usually makes little sense):
        
        >>> print(stream.select('.'))
        <doc><elem>foo</elem><elem>bar</elem></doc>
        
        :param path: a string containing the XPath expression
        :param namespaces: mapping of namespace prefixes used in the path
        :param variables: mapping of variable names to values
        :return: the selected substream
        :rtype: `Stream`
        :raises PathSyntaxError: if the given path expression is invalid or not
                                 supported
        """
        from genshi.path import Path
        return Path(path).select(self, namespaces, variables)

    def serialize(self, method='xml', **kwargs):
        """Generate strings corresponding to a specific serialization of the
        stream.
        
        Unlike the `render()` method, this method is a generator that returns
        the serialized output incrementally, as opposed to returning a single
        string.
        
        Any additional keyword arguments are passed to the serializer, and thus
        depend on the `method` parameter value.
        
        :param method: determines how the stream is serialized; can be either
                       "xml", "xhtml", "html", "text", or a custom serializer
                       class; if `None`, the default serialization method of
                       the stream is used
        :return: an iterator over the serialization results (`Markup` or
                 `unicode` objects, depending on the serialization method)
        :rtype: ``iterator``
        :see: XMLSerializer, XHTMLSerializer, HTMLSerializer, TextSerializer
        """
        from genshi.output import get_serializer
        if method is None:
            method = self.serializer or 'xml'
        return get_serializer(method, **kwargs)(_ensure(self))

    def __str__(self):
        return self.render()

    def __unicode__(self):
        return self.render(encoding=None)

    def __html__(self):
        return self