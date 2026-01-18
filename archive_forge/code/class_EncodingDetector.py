from html.entities import codepoint2name
from collections import defaultdict
import codecs
import re
import logging
import string
from html.entities import html5
class EncodingDetector:
    """Suggests a number of possible encodings for a bytestring.

    Order of precedence:

    1. Encodings you specifically tell EncodingDetector to try first
    (the known_definite_encodings argument to the constructor).

    2. An encoding determined by sniffing the document's byte-order mark.

    3. Encodings you specifically tell EncodingDetector to try if
    byte-order mark sniffing fails (the user_encodings argument to the
    constructor).

    4. An encoding declared within the bytestring itself, either in an
    XML declaration (if the bytestring is to be interpreted as an XML
    document), or in a <meta> tag (if the bytestring is to be
    interpreted as an HTML document.)

    5. An encoding detected through textual analysis by chardet,
    cchardet, or a similar external library.

    4. UTF-8.

    5. Windows-1252.

    """

    def __init__(self, markup, known_definite_encodings=None, is_html=False, exclude_encodings=None, user_encodings=None, override_encodings=None):
        """Constructor.

        :param markup: Some markup in an unknown encoding.

        :param known_definite_encodings: When determining the encoding
            of `markup`, these encodings will be tried first, in
            order. In HTML terms, this corresponds to the "known
            definite encoding" step defined here:
            https://html.spec.whatwg.org/multipage/parsing.html#parsing-with-a-known-character-encoding

        :param user_encodings: These encodings will be tried after the
            `known_definite_encodings` have been tried and failed, and
            after an attempt to sniff the encoding by looking at a
            byte order mark has failed. In HTML terms, this
            corresponds to the step "user has explicitly instructed
            the user agent to override the document's character
            encoding", defined here:
            https://html.spec.whatwg.org/multipage/parsing.html#determining-the-character-encoding

        :param override_encodings: A deprecated alias for
            known_definite_encodings. Any encodings here will be tried
            immediately after the encodings in
            known_definite_encodings.

        :param is_html: If True, this markup is considered to be
            HTML. Otherwise it's assumed to be XML.

        :param exclude_encodings: These encodings will not be tried,
            even if they otherwise would be.

        """
        self.known_definite_encodings = list(known_definite_encodings or [])
        if override_encodings:
            self.known_definite_encodings += override_encodings
        self.user_encodings = user_encodings or []
        exclude_encodings = exclude_encodings or []
        self.exclude_encodings = set([x.lower() for x in exclude_encodings])
        self.chardet_encoding = None
        self.is_html = is_html
        self.declared_encoding = None
        self.markup, self.sniffed_encoding = self.strip_byte_order_mark(markup)

    def _usable(self, encoding, tried):
        """Should we even bother to try this encoding?

        :param encoding: Name of an encoding.
        :param tried: Encodings that have already been tried. This will be modified
            as a side effect.
        """
        if encoding is not None:
            encoding = encoding.lower()
            if encoding in self.exclude_encodings:
                return False
            if encoding not in tried:
                tried.add(encoding)
                return True
        return False

    @property
    def encodings(self):
        """Yield a number of encodings that might work for this markup.

        :yield: A sequence of strings.
        """
        tried = set()
        for e in self.known_definite_encodings:
            if self._usable(e, tried):
                yield e
        if self._usable(self.sniffed_encoding, tried):
            yield self.sniffed_encoding
        for e in self.user_encodings:
            if self._usable(e, tried):
                yield e
        if self.declared_encoding is None:
            self.declared_encoding = self.find_declared_encoding(self.markup, self.is_html)
        if self._usable(self.declared_encoding, tried):
            yield self.declared_encoding
        if self.chardet_encoding is None:
            self.chardet_encoding = chardet_dammit(self.markup)
        if self._usable(self.chardet_encoding, tried):
            yield self.chardet_encoding
        for e in ('utf-8', 'windows-1252'):
            if self._usable(e, tried):
                yield e

    @classmethod
    def strip_byte_order_mark(cls, data):
        """If a byte-order mark is present, strip it and return the encoding it implies.

        :param data: Some markup.
        :return: A 2-tuple (modified data, implied encoding)
        """
        encoding = None
        if isinstance(data, str):
            return (data, encoding)
        if len(data) >= 4 and data[:2] == b'\xfe\xff' and (data[2:4] != '\x00\x00'):
            encoding = 'utf-16be'
            data = data[2:]
        elif len(data) >= 4 and data[:2] == b'\xff\xfe' and (data[2:4] != '\x00\x00'):
            encoding = 'utf-16le'
            data = data[2:]
        elif data[:3] == b'\xef\xbb\xbf':
            encoding = 'utf-8'
            data = data[3:]
        elif data[:4] == b'\x00\x00\xfe\xff':
            encoding = 'utf-32be'
            data = data[4:]
        elif data[:4] == b'\xff\xfe\x00\x00':
            encoding = 'utf-32le'
            data = data[4:]
        return (data, encoding)

    @classmethod
    def find_declared_encoding(cls, markup, is_html=False, search_entire_document=False):
        """Given a document, tries to find its declared encoding.

        An XML encoding is declared at the beginning of the document.

        An HTML encoding is declared in a <meta> tag, hopefully near the
        beginning of the document.

        :param markup: Some markup.
        :param is_html: If True, this markup is considered to be HTML. Otherwise
            it's assumed to be XML.
        :param search_entire_document: Since an encoding is supposed to declared near the beginning
            of the document, most of the time it's only necessary to search a few kilobytes of data.
            Set this to True to force this method to search the entire document.
        """
        if search_entire_document:
            xml_endpos = html_endpos = len(markup)
        else:
            xml_endpos = 1024
            html_endpos = max(2048, int(len(markup) * 0.05))
        if isinstance(markup, bytes):
            res = encoding_res[bytes]
        else:
            res = encoding_res[str]
        xml_re = res['xml']
        html_re = res['html']
        declared_encoding = None
        declared_encoding_match = xml_re.search(markup, endpos=xml_endpos)
        if not declared_encoding_match and is_html:
            declared_encoding_match = html_re.search(markup, endpos=html_endpos)
        if declared_encoding_match is not None:
            declared_encoding = declared_encoding_match.groups()[0]
        if declared_encoding:
            if isinstance(declared_encoding, bytes):
                declared_encoding = declared_encoding.decode('ascii', 'replace')
            return declared_encoding.lower()
        return None