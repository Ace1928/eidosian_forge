from itertools import chain
import re
import six
from genshi.core import escape, Attrs, Markup, QName, StreamEventKind
from genshi.core import START, END, TEXT, XML_DECL, DOCTYPE, START_NS, END_NS, \
class WhitespaceFilter(object):
    """A filter that removes extraneous ignorable white space from the
    stream.
    """

    def __init__(self, preserve=None, noescape=None):
        """Initialize the filter.
        
        :param preserve: a set or sequence of tag names for which white-space
                         should be preserved
        :param noescape: a set or sequence of tag names for which text content
                         should not be escaped
        
        The `noescape` set is expected to refer to elements that cannot contain
        further child elements (such as ``<style>`` or ``<script>`` in HTML
        documents).
        """
        if preserve is None:
            preserve = []
        self.preserve = frozenset(preserve)
        if noescape is None:
            noescape = []
        self.noescape = frozenset(noescape)

    def __call__(self, stream, ctxt=None, space=XML_NAMESPACE['space'], trim_trailing_space=re.compile('[ \t]+(?=\n)').sub, collapse_lines=re.compile('\n{2,}').sub):
        mjoin = Markup('').join
        preserve_elems = self.preserve
        preserve = 0
        noescape_elems = self.noescape
        noescape = False
        textbuf = []
        push_text = textbuf.append
        pop_text = textbuf.pop
        for kind, data, pos in chain(stream, [(None, None, None)]):
            if kind is TEXT:
                if noescape:
                    data = Markup(data)
                push_text(data)
            else:
                if textbuf:
                    if len(textbuf) > 1:
                        text = mjoin(textbuf, escape_quotes=False)
                        del textbuf[:]
                    else:
                        text = escape(pop_text(), quotes=False)
                    if not preserve:
                        text = collapse_lines('\n', trim_trailing_space('', text))
                    yield (TEXT, Markup(text), pos)
                if kind is START:
                    tag, attrs = data
                    if preserve or (tag in preserve_elems or attrs.get(space) == 'preserve'):
                        preserve += 1
                    if not noescape and tag in noescape_elems:
                        noescape = True
                elif kind is END:
                    noescape = False
                    if preserve:
                        preserve -= 1
                elif kind is START_CDATA:
                    noescape = True
                elif kind is END_CDATA:
                    noescape = False
                if kind:
                    yield (kind, data, pos)