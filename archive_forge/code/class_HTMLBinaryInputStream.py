from __future__ import absolute_import, division, unicode_literals
from six import text_type
from six.moves import http_client, urllib
import codecs
import re
from io import BytesIO, StringIO
from tensorboard._vendor import webencodings
from .constants import EOF, spaceCharacters, asciiLetters, asciiUppercase
from .constants import _ReparseException
from . import _utils
class HTMLBinaryInputStream(HTMLUnicodeInputStream):
    """Provides a unicode stream of characters to the HTMLTokenizer.

    This class takes care of character encoding and removing or replacing
    incorrect byte-sequences and also provides column and line tracking.

    """

    def __init__(self, source, override_encoding=None, transport_encoding=None, same_origin_parent_encoding=None, likely_encoding=None, default_encoding='windows-1252', useChardet=True):
        """Initialises the HTMLInputStream.

        HTMLInputStream(source, [encoding]) -> Normalized stream from source
        for use by html5lib.

        source can be either a file-object, local filename or a string.

        The optional encoding parameter must be a string that indicates
        the encoding.  If specified, that encoding will be used,
        regardless of any BOM or later declaration (such as in a meta
        element)

        """
        self.rawStream = self.openStream(source)
        HTMLUnicodeInputStream.__init__(self, self.rawStream)
        self.numBytesMeta = 1024
        self.numBytesChardet = 100
        self.override_encoding = override_encoding
        self.transport_encoding = transport_encoding
        self.same_origin_parent_encoding = same_origin_parent_encoding
        self.likely_encoding = likely_encoding
        self.default_encoding = default_encoding
        self.charEncoding = self.determineEncoding(useChardet)
        assert self.charEncoding[0] is not None
        self.reset()

    def reset(self):
        self.dataStream = self.charEncoding[0].codec_info.streamreader(self.rawStream, 'replace')
        HTMLUnicodeInputStream.reset(self)

    def openStream(self, source):
        """Produces a file object from source.

        source can be either a file object, local filename or a string.

        """
        if hasattr(source, 'read'):
            stream = source
        else:
            stream = BytesIO(source)
        try:
            stream.seek(stream.tell())
        except Exception:
            stream = BufferedStream(stream)
        return stream

    def determineEncoding(self, chardet=True):
        charEncoding = (self.detectBOM(), 'certain')
        if charEncoding[0] is not None:
            return charEncoding
        charEncoding = (lookupEncoding(self.override_encoding), 'certain')
        if charEncoding[0] is not None:
            return charEncoding
        charEncoding = (lookupEncoding(self.transport_encoding), 'certain')
        if charEncoding[0] is not None:
            return charEncoding
        charEncoding = (self.detectEncodingMeta(), 'tentative')
        if charEncoding[0] is not None:
            return charEncoding
        charEncoding = (lookupEncoding(self.same_origin_parent_encoding), 'tentative')
        if charEncoding[0] is not None and (not charEncoding[0].name.startswith('utf-16')):
            return charEncoding
        charEncoding = (lookupEncoding(self.likely_encoding), 'tentative')
        if charEncoding[0] is not None:
            return charEncoding
        if chardet:
            try:
                from chardet.universaldetector import UniversalDetector
            except ImportError:
                pass
            else:
                buffers = []
                detector = UniversalDetector()
                while not detector.done:
                    buffer = self.rawStream.read(self.numBytesChardet)
                    assert isinstance(buffer, bytes)
                    if not buffer:
                        break
                    buffers.append(buffer)
                    detector.feed(buffer)
                detector.close()
                encoding = lookupEncoding(detector.result['encoding'])
                self.rawStream.seek(0)
                if encoding is not None:
                    return (encoding, 'tentative')
        charEncoding = (lookupEncoding(self.default_encoding), 'tentative')
        if charEncoding[0] is not None:
            return charEncoding
        return (lookupEncoding('windows-1252'), 'tentative')

    def changeEncoding(self, newEncoding):
        assert self.charEncoding[1] != 'certain'
        newEncoding = lookupEncoding(newEncoding)
        if newEncoding is None:
            return
        if newEncoding.name in ('utf-16be', 'utf-16le'):
            newEncoding = lookupEncoding('utf-8')
            assert newEncoding is not None
        elif newEncoding == self.charEncoding[0]:
            self.charEncoding = (self.charEncoding[0], 'certain')
        else:
            self.rawStream.seek(0)
            self.charEncoding = (newEncoding, 'certain')
            self.reset()
            raise _ReparseException('Encoding changed from %s to %s' % (self.charEncoding[0], newEncoding))

    def detectBOM(self):
        """Attempts to detect at BOM at the start of the stream. If
        an encoding can be determined from the BOM return the name of the
        encoding otherwise return None"""
        bomDict = {codecs.BOM_UTF8: 'utf-8', codecs.BOM_UTF16_LE: 'utf-16le', codecs.BOM_UTF16_BE: 'utf-16be', codecs.BOM_UTF32_LE: 'utf-32le', codecs.BOM_UTF32_BE: 'utf-32be'}
        string = self.rawStream.read(4)
        assert isinstance(string, bytes)
        encoding = bomDict.get(string[:3])
        seek = 3
        if not encoding:
            encoding = bomDict.get(string)
            seek = 4
            if not encoding:
                encoding = bomDict.get(string[:2])
                seek = 2
        if encoding:
            self.rawStream.seek(seek)
            return lookupEncoding(encoding)
        else:
            self.rawStream.seek(0)
            return None

    def detectEncodingMeta(self):
        """Report the encoding declared by the meta element
        """
        buffer = self.rawStream.read(self.numBytesMeta)
        assert isinstance(buffer, bytes)
        parser = EncodingParser(buffer)
        self.rawStream.seek(0)
        encoding = parser.getEncoding()
        if encoding is not None and encoding.name in ('utf-16be', 'utf-16le'):
            encoding = lookupEncoding('utf-8')
        return encoding