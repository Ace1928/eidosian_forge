from itertools import chain
import re
import six
from genshi.core import escape, Attrs, Markup, QName, StreamEventKind
from genshi.core import START, END, TEXT, XML_DECL, DOCTYPE, START_NS, END_NS, \
class TextSerializer(object):
    """Produces plain text from an event stream.
    
    Only text events are included in the output. Unlike the other serializer,
    special XML characters are not escaped:
    
    >>> from genshi.builder import tag
    >>> elem = tag.div(tag.a('<Hello!>', href='foo'), tag.br)
    >>> print(elem)
    <div><a href="foo">&lt;Hello!&gt;</a><br/></div>
    >>> print(''.join(TextSerializer()(elem.generate())))
    <Hello!>

    If text events contain literal markup (instances of the `Markup` class),
    that markup is by default passed through unchanged:
    
    >>> elem = tag.div(Markup('<a href="foo">Hello &amp; Bye!</a><br/>'))
    >>> print(elem.generate().render(TextSerializer, encoding=None))
    <a href="foo">Hello &amp; Bye!</a><br/>
    
    You can use the ``strip_markup`` to change this behavior, so that tags and
    entities are stripped from the output (or in the case of entities,
    replaced with the equivalent character):

    >>> print(elem.generate().render(TextSerializer, strip_markup=True,
    ...                              encoding=None))
    Hello & Bye!
    """

    def __init__(self, strip_markup=False):
        """Create the serializer.
        
        :param strip_markup: whether markup (tags and encoded characters) found
                             in the text should be removed
        """
        self.strip_markup = strip_markup

    def __call__(self, stream):
        strip_markup = self.strip_markup
        for event in stream:
            if event[0] is TEXT:
                data = event[1]
                if strip_markup and type(data) is Markup:
                    data = data.striptags().stripentities()
                yield six.text_type(data)