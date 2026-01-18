import textwrap
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tag import map_tag
from nltk.tree import Tree
from nltk.util import LazyConcatenation, LazyMap
def _grids(self, fileids=None):
    return concat([StreamBackedCorpusView(fileid, self._read_grid_block, encoding=enc) for fileid, enc in self.abspaths(fileids, True)])