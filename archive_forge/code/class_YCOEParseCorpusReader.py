import os
import re
from nltk.corpus.reader.api import *
from nltk.corpus.reader.bracket_parse import BracketParseCorpusReader
from nltk.corpus.reader.tagged import TaggedCorpusReader
from nltk.corpus.reader.util import *
from nltk.tokenize import RegexpTokenizer
class YCOEParseCorpusReader(BracketParseCorpusReader):
    """Specialized version of the standard bracket parse corpus reader
    that strips out (CODE ...) and (ID ...) nodes."""

    def _parse(self, t):
        t = re.sub('(?u)\\((CODE|ID)[^\\)]*\\)', '', t)
        if re.match('\\s*\\(\\s*\\)\\s*$', t):
            return None
        return BracketParseCorpusReader._parse(self, t)