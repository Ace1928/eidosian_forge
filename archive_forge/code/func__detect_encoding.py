import codecs
from xml.etree import ElementTree
from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.util import *
from nltk.data import SeekableUnicodeStreamReader
from nltk.internals import ElementWrapper
from nltk.tokenize import WordPunctTokenizer
def _detect_encoding(self, fileid):
    if isinstance(fileid, PathPointer):
        try:
            infile = fileid.open()
            s = infile.readline()
        finally:
            infile.close()
    else:
        with open(fileid, 'rb') as infile:
            s = infile.readline()
    if s.startswith(codecs.BOM_UTF16_BE):
        return 'utf-16-be'
    if s.startswith(codecs.BOM_UTF16_LE):
        return 'utf-16-le'
    if s.startswith(codecs.BOM_UTF32_BE):
        return 'utf-32-be'
    if s.startswith(codecs.BOM_UTF32_LE):
        return 'utf-32-le'
    if s.startswith(codecs.BOM_UTF8):
        return 'utf-8'
    m = re.match(b'\\s*<\\?xml\\b.*\\bencoding="([^"]+)"', s)
    if m:
        return m.group(1).decode()
    m = re.match(b"\\s*<\\?xml\\b.*\\bencoding='([^']+)'", s)
    if m:
        return m.group(1).decode()
    return 'utf-8'