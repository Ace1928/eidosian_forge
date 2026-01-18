import re
import six
from genshi.core import Attrs, QName, stripentities
from genshi.core import END, START, TEXT, COMMENT
class HTMLSanitizer(object):
    """A filter that removes potentially dangerous HTML tags and attributes
    from the stream.
    
    >>> from genshi import HTML
    >>> html = HTML('<div><script>alert(document.cookie)</script></div>', encoding='utf-8')
    >>> print(html | HTMLSanitizer())
    <div/>
    
    The default set of safe tags and attributes can be modified when the filter
    is instantiated. For example, to allow inline ``style`` attributes, the
    following instantation would work:
    
    >>> html = HTML('<div style="background: #000"></div>', encoding='utf-8')
    >>> sanitizer = HTMLSanitizer(safe_attrs=HTMLSanitizer.SAFE_ATTRS | set(['style']))
    >>> print(html | sanitizer)
    <div style="background: #000"/>
    
    Note that even in this case, the filter *does* attempt to remove dangerous
    constructs from style attributes:

    >>> html = HTML('<div style="background: url(javascript:void); color: #000"></div>', encoding='utf-8')
    >>> print(html | sanitizer)
    <div style="color: #000"/>
    
    This handles HTML entities, unicode escapes in CSS and Javascript text, as
    well as a lot of other things. However, the style tag is still excluded by
    default because it is very hard for such sanitizing to be completely safe,
    especially considering how much error recovery current web browsers perform.
    
    It also does some basic filtering of CSS properties that may be used for
    typical phishing attacks. For more sophisticated filtering, this class
    provides a couple of hooks that can be overridden in sub-classes.
    
    :warn: Note that this special processing of CSS is currently only applied to
           style attributes, **not** style elements.
    """
    SAFE_TAGS = frozenset(['a', 'abbr', 'acronym', 'address', 'area', 'b', 'big', 'blockquote', 'br', 'button', 'caption', 'center', 'cite', 'code', 'col', 'colgroup', 'dd', 'del', 'dfn', 'dir', 'div', 'dl', 'dt', 'em', 'fieldset', 'font', 'form', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'hr', 'i', 'img', 'input', 'ins', 'kbd', 'label', 'legend', 'li', 'map', 'menu', 'ol', 'optgroup', 'option', 'p', 'pre', 'q', 's', 'samp', 'select', 'small', 'span', 'strike', 'strong', 'sub', 'sup', 'table', 'tbody', 'td', 'textarea', 'tfoot', 'th', 'thead', 'tr', 'tt', 'u', 'ul', 'var'])
    SAFE_ATTRS = frozenset(['abbr', 'accept', 'accept-charset', 'accesskey', 'action', 'align', 'alt', 'axis', 'bgcolor', 'border', 'cellpadding', 'cellspacing', 'char', 'charoff', 'charset', 'checked', 'cite', 'class', 'clear', 'cols', 'colspan', 'color', 'compact', 'coords', 'datetime', 'dir', 'disabled', 'enctype', 'for', 'frame', 'headers', 'height', 'href', 'hreflang', 'hspace', 'id', 'ismap', 'label', 'lang', 'longdesc', 'maxlength', 'media', 'method', 'multiple', 'name', 'nohref', 'noshade', 'nowrap', 'prompt', 'readonly', 'rel', 'rev', 'rows', 'rowspan', 'rules', 'scope', 'selected', 'shape', 'size', 'span', 'src', 'start', 'summary', 'tabindex', 'target', 'title', 'type', 'usemap', 'valign', 'value', 'vspace', 'width'])
    SAFE_CSS = frozenset(['background', 'background-attachment', 'background-color', 'background-image', 'background-position', 'background-repeat', 'border', 'border-bottom', 'border-bottom-color', 'border-bottom-style', 'border-bottom-width', 'border-collapse', 'border-color', 'border-left', 'border-left-color', 'border-left-style', 'border-left-width', 'border-right', 'border-right-color', 'border-right-style', 'border-right-width', 'border-spacing', 'border-style', 'border-top', 'border-top-color', 'border-top-style', 'border-top-width', 'border-width', 'bottom', 'caption-side', 'clear', 'clip', 'color', 'content', 'counter-increment', 'counter-reset', 'cursor', 'direction', 'display', 'empty-cells', 'float', 'font', 'font-family', 'font-size', 'font-style', 'font-variant', 'font-weight', 'height', 'left', 'letter-spacing', 'line-height', 'list-style', 'list-style-image', 'list-style-position', 'list-style-type', 'margin', 'margin-bottom', 'margin-left', 'margin-right', 'margin-top', 'max-height', 'max-width', 'min-height', 'min-width', 'opacity', 'orphans', 'outline', 'outline-color', 'outline-style', 'outline-width', 'overflow', 'padding', 'padding-bottom', 'padding-left', 'padding-right', 'padding-top', 'page-break-after', 'page-break-before', 'page-break-inside', 'quotes', 'right', 'table-layout', 'text-align', 'text-decoration', 'text-indent', 'text-transform', 'top', 'unicode-bidi', 'vertical-align', 'visibility', 'white-space', 'widows', 'width', 'word-spacing', 'z-index'])
    SAFE_SCHEMES = frozenset(['file', 'ftp', 'http', 'https', 'mailto', None])
    URI_ATTRS = frozenset(['action', 'background', 'dynsrc', 'href', 'lowsrc', 'src'])

    def __init__(self, safe_tags=SAFE_TAGS, safe_attrs=SAFE_ATTRS, safe_schemes=SAFE_SCHEMES, uri_attrs=URI_ATTRS, safe_css=SAFE_CSS):
        """Create the sanitizer.
        
        The exact set of allowed elements and attributes can be configured.
        
        :param safe_tags: a set of tag names that are considered safe
        :param safe_attrs: a set of attribute names that are considered safe
        :param safe_schemes: a set of URI schemes that are considered safe
        :param uri_attrs: a set of names of attributes that contain URIs
        """
        self.safe_tags = safe_tags
        self.safe_attrs = safe_attrs
        self.safe_css = safe_css
        self.uri_attrs = uri_attrs
        self.safe_schemes = safe_schemes
    _EXPRESSION_SEARCH = re.compile(u'\n        [eE\n         Ｅ # FULLWIDTH LATIN CAPITAL LETTER E\n         ｅ # FULLWIDTH LATIN SMALL LETTER E\n        ]\n        [xX\n         Ｘ # FULLWIDTH LATIN CAPITAL LETTER X\n         ｘ # FULLWIDTH LATIN SMALL LETTER X\n        ]\n        [pP\n         Ｐ # FULLWIDTH LATIN CAPITAL LETTER P\n         ｐ # FULLWIDTH LATIN SMALL LETTER P\n        ]\n        [rR\n         ʀ # LATIN LETTER SMALL CAPITAL R\n         Ｒ # FULLWIDTH LATIN CAPITAL LETTER R\n         ｒ # FULLWIDTH LATIN SMALL LETTER R\n        ]\n        [eE\n         Ｅ # FULLWIDTH LATIN CAPITAL LETTER E\n         ｅ # FULLWIDTH LATIN SMALL LETTER E\n        ]\n        [sS\n         Ｓ # FULLWIDTH LATIN CAPITAL LETTER S\n         ｓ # FULLWIDTH LATIN SMALL LETTER S\n        ]{2}\n        [iI\n         ɪ # LATIN LETTER SMALL CAPITAL I\n         Ｉ # FULLWIDTH LATIN CAPITAL LETTER I\n         ｉ # FULLWIDTH LATIN SMALL LETTER I\n        ]\n        [oO\n         Ｏ # FULLWIDTH LATIN CAPITAL LETTER O\n         ｏ # FULLWIDTH LATIN SMALL LETTER O\n        ]\n        [nN\n         ɴ # LATIN LETTER SMALL CAPITAL N\n         Ｎ # FULLWIDTH LATIN CAPITAL LETTER N\n         ｎ # FULLWIDTH LATIN SMALL LETTER N\n        ]\n        ', re.VERBOSE).search
    _URL_FINDITER = re.compile(u'[Uu][Rrʀ][Llʟ]%s*\\(([^)]+)' % '\\s').finditer

    def __call__(self, stream):
        """Apply the filter to the given stream.
        
        :param stream: the markup event stream to filter
        """
        waiting_for = None
        for kind, data, pos in stream:
            if kind is START:
                if waiting_for:
                    continue
                tag, attrs = data
                if not self.is_safe_elem(tag, attrs):
                    waiting_for = tag
                    continue
                new_attrs = []
                for attr, value in attrs:
                    value = stripentities(value)
                    if attr not in self.safe_attrs:
                        continue
                    elif attr in self.uri_attrs:
                        if not self.is_safe_uri(value):
                            continue
                    elif attr == 'style':
                        decls = self.sanitize_css(value)
                        if not decls:
                            continue
                        value = '; '.join(decls)
                    new_attrs.append((attr, value))
                yield (kind, (tag, Attrs(new_attrs)), pos)
            elif kind is END:
                tag = data
                if waiting_for:
                    if waiting_for == tag:
                        waiting_for = None
                else:
                    yield (kind, data, pos)
            elif kind is not COMMENT:
                if not waiting_for:
                    yield (kind, data, pos)

    def is_safe_css(self, propname, value):
        """Determine whether the given css property declaration is to be
        considered safe for inclusion in the output.
        
        :param propname: the CSS property name
        :param value: the value of the property
        :return: whether the property value should be considered safe
        :rtype: bool
        :since: version 0.6
        """
        if propname not in self.safe_css:
            return False
        if propname.startswith('margin') and '-' in value:
            return False
        return True

    def is_safe_elem(self, tag, attrs):
        """Determine whether the given element should be considered safe for
        inclusion in the output.
        
        :param tag: the tag name of the element
        :type tag: QName
        :param attrs: the element attributes
        :type attrs: Attrs
        :return: whether the element should be considered safe
        :rtype: bool
        :since: version 0.6
        """
        if tag not in self.safe_tags:
            return False
        if tag.localname == 'input':
            input_type = attrs.get('type', '').lower()
            if input_type == 'password':
                return False
        return True

    def is_safe_uri(self, uri):
        """Determine whether the given URI is to be considered safe for
        inclusion in the output.
        
        The default implementation checks whether the scheme of the URI is in
        the set of allowed URIs (`safe_schemes`).
        
        >>> sanitizer = HTMLSanitizer()
        >>> sanitizer.is_safe_uri('http://example.org/')
        True
        >>> sanitizer.is_safe_uri('javascript:alert(document.cookie)')
        False
        
        :param uri: the URI to check
        :return: `True` if the URI can be considered safe, `False` otherwise
        :rtype: `bool`
        :since: version 0.4.3
        """
        if '#' in uri:
            uri = uri.split('#', 1)[0]
        if ':' not in uri:
            return True
        chars = [char for char in uri.split(':', 1)[0] if char.isalnum()]
        return ''.join(chars).lower() in self.safe_schemes

    def sanitize_css(self, text):
        """Remove potentially dangerous property declarations from CSS code.
        
        In particular, properties using the CSS ``url()`` function with a scheme
        that is not considered safe are removed:
        
        >>> sanitizer = HTMLSanitizer()
        >>> sanitizer.sanitize_css(u'''
        ...   background: url(javascript:alert("foo"));
        ...   color: #000;
        ... ''')
        ['color: #000']
        
        Also, the proprietary Internet Explorer function ``expression()`` is
        always stripped:
        
        >>> sanitizer.sanitize_css(u'''
        ...   background: #fff;
        ...   color: #000;
        ...   width: e/**/xpression(alert("foo"));
        ... ''')
        ['background: #fff', 'color: #000']
        
        :param text: the CSS text; this is expected to be `unicode` and to not
                     contain any character or numeric references
        :return: a list of declarations that are considered safe
        :rtype: `list`
        :since: version 0.4.3
        """
        decls = []
        text = self._strip_css_comments(self._replace_unicode_escapes(text))
        for decl in text.split(';'):
            decl = decl.strip()
            if not decl:
                continue
            try:
                propname, value = decl.split(':', 1)
            except ValueError:
                continue
            if not self.is_safe_css(propname.strip().lower(), value.strip()):
                continue
            is_evil = False
            if self._EXPRESSION_SEARCH(value):
                is_evil = True
            for match in self._URL_FINDITER(value):
                if not self.is_safe_uri(match.group(1)):
                    is_evil = True
                    break
            if not is_evil:
                decls.append(decl.strip())
        return decls
    _NORMALIZE_NEWLINES = re.compile('\\r\\n').sub
    _UNICODE_ESCAPE = re.compile('\\\\([0-9a-fA-F]{1,6})\\s?|\\\\([^\\r\\n\\f0-9a-fA-F\'"{};:()#*])', re.UNICODE).sub

    def _replace_unicode_escapes(self, text):

        def _repl(match):
            t = match.group(1)
            if t:
                return six.unichr(int(t, 16))
            t = match.group(2)
            if t == '\\':
                return '\\\\'
            else:
                return t
        return self._UNICODE_ESCAPE(_repl, self._NORMALIZE_NEWLINES('\n', text))
    _CSS_COMMENTS = re.compile('/\\*.*?\\*/').sub

    def _strip_css_comments(self, text):
        return self._CSS_COMMENTS('', text)