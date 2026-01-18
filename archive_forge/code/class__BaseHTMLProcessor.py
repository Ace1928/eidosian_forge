import html.entities
import re
from .sgml import *
class _BaseHTMLProcessor(sgmllib.SGMLParser, object):
    special = re.compile('[<>\'"]')
    bare_ampersand = re.compile('&(?!#\\d+;|#x[0-9a-fA-F]+;|\\w+;)')
    elements_no_end_tag = {'area', 'base', 'basefont', 'br', 'col', 'command', 'embed', 'frame', 'hr', 'img', 'input', 'isindex', 'keygen', 'link', 'meta', 'param', 'source', 'track', 'wbr'}

    def __init__(self, encoding=None, _type='application/xhtml+xml'):
        if encoding:
            self.encoding = encoding
        self._type = _type
        self.pieces = []
        super(_BaseHTMLProcessor, self).__init__()

    def reset(self):
        self.pieces = []
        super(_BaseHTMLProcessor, self).reset()

    def _shorttag_replace(self, match):
        """
        :type match: Match[str]
        :rtype: str
        """
        tag = match.group(1)
        if tag in self.elements_no_end_tag:
            return '<' + tag + ' />'
        else:
            return '<' + tag + '></' + tag + '>'

    def goahead(self, i):
        raise NotImplementedError
    try:
        goahead.__code__ = sgmllib.SGMLParser.goahead.__code__
    except AttributeError:
        goahead.func_code = sgmllib.SGMLParser.goahead.func_code

    def __parse_starttag(self, i):
        raise NotImplementedError
    try:
        __parse_starttag.__code__ = sgmllib.SGMLParser.parse_starttag.__code__
    except AttributeError:
        __parse_starttag.func_code = sgmllib.SGMLParser.parse_starttag.func_code

    def parse_starttag(self, i):
        j = self.__parse_starttag(i)
        if self._type == 'application/xhtml+xml':
            if j > 2 and self.rawdata[j - 2:j] == '/>':
                self.unknown_endtag(self.lasttag)
        return j

    def feed(self, data):
        """
        :type data: str
        :rtype: None
        """
        data = re.sub('<!((?!DOCTYPE|--|\\[))', '&lt;!\\1', data, re.IGNORECASE)
        data = re.sub('<([^<>\\s]+?)\\s*/>', self._shorttag_replace, data)
        data = data.replace('&#39;', "'")
        data = data.replace('&#34;', '"')
        super(_BaseHTMLProcessor, self).feed(data)
        super(_BaseHTMLProcessor, self).close()

    @staticmethod
    def normalize_attrs(attrs):
        """
        :type attrs: List[Tuple[str, str]]
        :rtype: List[Tuple[str, str]]
        """
        if not attrs:
            return attrs
        attrs_d = {k.lower(): v for k, v in attrs}
        attrs = [(k, k in ('rel', 'type') and v.lower() or v) for k, v in attrs_d.items()]
        attrs.sort()
        return attrs

    def unknown_starttag(self, tag, attrs):
        """
        :type tag: str
        :type attrs: List[Tuple[str, str]]
        :rtype: None
        """
        uattrs = []
        strattrs = ''
        if attrs:
            for key, value in attrs:
                value = value.replace('>', '&gt;')
                value = value.replace('<', '&lt;')
                value = value.replace('"', '&quot;')
                value = self.bare_ampersand.sub('&amp;', value)
                uattrs.append((key, value))
            strattrs = ''.join((' %s="%s"' % (key, value) for key, value in uattrs))
        if tag in self.elements_no_end_tag:
            self.pieces.append('<%s%s />' % (tag, strattrs))
        else:
            self.pieces.append('<%s%s>' % (tag, strattrs))

    def unknown_endtag(self, tag):
        """
        :type tag: str
        :rtype: None
        """
        if tag not in self.elements_no_end_tag:
            self.pieces.append('</%s>' % tag)

    def handle_charref(self, ref):
        """
        :type ref: str
        :rtype: None
        """
        ref = ref.lower()
        if ref.startswith('x'):
            value = int(ref[1:], 16)
        else:
            value = int(ref)
        if value in _cp1252:
            self.pieces.append('&#%s;' % hex(ord(_cp1252[value]))[1:])
        else:
            self.pieces.append('&#%s;' % ref)

    def handle_entityref(self, ref):
        """
        :type ref: str
        :rtype: None
        """
        if ref in html.entities.name2codepoint or ref == 'apos':
            self.pieces.append('&%s;' % ref)
        else:
            self.pieces.append('&amp;%s' % ref)

    def handle_data(self, text):
        """
        :type text: str
        :rtype: None
        """
        self.pieces.append(text)

    def handle_comment(self, text):
        """
        :type text: str
        :rtype: None
        """
        self.pieces.append('<!--%s-->' % text)

    def handle_pi(self, text):
        """
        :type text: str
        :rtype: None
        """
        self.pieces.append('<?%s>' % text)

    def handle_decl(self, text):
        """
        :type text: str
        :rtype: None
        """
        self.pieces.append('<!%s>' % text)
    _new_declname_match = re.compile('[a-zA-Z][-_.a-zA-Z0-9:]*\\s*').match

    def _scan_name(self, i, declstartpos):
        """
        :type i: int
        :type declstartpos: int
        :rtype: Tuple[Optional[str], int]
        """
        rawdata = self.rawdata
        n = len(rawdata)
        if i == n:
            return (None, -1)
        m = self._new_declname_match(rawdata, i)
        if m:
            s = m.group()
            name = s.strip()
            if i + len(s) == n:
                return (None, -1)
            return (name.lower(), m.end())
        else:
            self.handle_data(rawdata)
            return (None, -1)

    @staticmethod
    def convert_charref(name):
        """
        :type name: str
        :rtype: str
        """
        return '&#%s;' % name

    @staticmethod
    def convert_entityref(name):
        """
        :type name: str
        :rtype: str
        """
        return '&%s;' % name

    def output(self):
        """Return processed HTML as a single string.

        :rtype: str
        """
        return ''.join(self.pieces)

    def parse_declaration(self, i):
        """
        :type i: int
        :rtype: int
        """
        try:
            return sgmllib.SGMLParser.parse_declaration(self, i)
        except sgmllib.SGMLParseError:
            self.handle_data('&lt;')
            return i + 1