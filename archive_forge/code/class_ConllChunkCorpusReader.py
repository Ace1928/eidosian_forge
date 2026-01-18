import textwrap
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tag import map_tag
from nltk.tree import Tree
from nltk.util import LazyConcatenation, LazyMap
class ConllChunkCorpusReader(ConllCorpusReader):
    """
    A ConllCorpusReader whose data file contains three columns: words,
    pos, and chunk.
    """

    def __init__(self, root, fileids, chunk_types, encoding='utf8', tagset=None, separator=None):
        ConllCorpusReader.__init__(self, root, fileids, ('words', 'pos', 'chunk'), chunk_types=chunk_types, encoding=encoding, tagset=tagset, separator=separator)