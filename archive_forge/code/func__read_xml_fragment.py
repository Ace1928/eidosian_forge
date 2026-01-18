import codecs
from xml.etree import ElementTree
from nltk.corpus.reader.api import CorpusReader
from nltk.corpus.reader.util import *
from nltk.data import SeekableUnicodeStreamReader
from nltk.internals import ElementWrapper
from nltk.tokenize import WordPunctTokenizer
def _read_xml_fragment(self, stream):
    """
        Read a string from the given stream that does not contain any
        un-closed tags.  In particular, this function first reads a
        block from the stream of size ``self._BLOCK_SIZE``.  It then
        checks if that block contains an un-closed tag.  If it does,
        then this function either backtracks to the last '<', or reads
        another block.
        """
    fragment = ''
    if isinstance(stream, SeekableUnicodeStreamReader):
        startpos = stream.tell()
    while True:
        xml_block = stream.read(self._BLOCK_SIZE)
        fragment += xml_block
        if self._VALID_XML_RE.match(fragment):
            return fragment
        if re.search('[<>]', fragment).group(0) == '>':
            pos = stream.tell() - (len(fragment) - re.search('[<>]', fragment).end())
            raise ValueError('Unexpected ">" near char %s' % pos)
        if not xml_block:
            raise ValueError('Unexpected end of file: tag not closed')
        last_open_bracket = fragment.rfind('<')
        if last_open_bracket > 0:
            if self._VALID_XML_RE.match(fragment[:last_open_bracket]):
                if isinstance(stream, SeekableUnicodeStreamReader):
                    stream.seek(startpos)
                    stream.char_seek_forward(last_open_bracket)
                else:
                    stream.seek(-(len(fragment) - last_open_bracket), 1)
                return fragment[:last_open_bracket]