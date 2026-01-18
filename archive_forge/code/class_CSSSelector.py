from . import etree
class CSSSelector(etree.XPath):
    """A CSS selector.

    Usage::

        >>> from lxml import etree, cssselect
        >>> select = cssselect.CSSSelector("a tag > child")

        >>> root = etree.XML("<a><b><c/><tag><child>TEXT</child></tag></b></a>")
        >>> [ el.tag for el in select(root) ]
        ['child']

    To use CSS namespaces, you need to pass a prefix-to-namespace
    mapping as ``namespaces`` keyword argument::

        >>> rdfns = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'
        >>> select_ns = cssselect.CSSSelector('root > rdf|Description',
        ...                                   namespaces={'rdf': rdfns})

        >>> rdf = etree.XML((
        ...     '<root xmlns:rdf="%s">'
        ...       '<rdf:Description>blah</rdf:Description>'
        ...     '</root>') % rdfns)
        >>> [(el.tag, el.text) for el in select_ns(rdf)]
        [('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description', 'blah')]

    """

    def __init__(self, css, namespaces=None, translator='xml'):
        if translator == 'xml':
            translator = LxmlTranslator()
        elif translator == 'html':
            translator = LxmlHTMLTranslator()
        elif translator == 'xhtml':
            translator = LxmlHTMLTranslator(xhtml=True)
        path = translator.css_to_xpath(css)
        super().__init__(path, namespaces=namespaces)
        self.css = css

    def __repr__(self):
        return '<%s %x for %r>' % (self.__class__.__name__, abs(id(self)), self.css)