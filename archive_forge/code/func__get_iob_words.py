import textwrap
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tag import map_tag
from nltk.tree import Tree
from nltk.util import LazyConcatenation, LazyMap
def _get_iob_words(self, grid, tagset=None):
    pos_tags = self._get_column(grid, self._colmap['pos'])
    if tagset and tagset != self._tagset:
        pos_tags = [map_tag(self._tagset, tagset, t) for t in pos_tags]
    return list(zip(self._get_column(grid, self._colmap['words']), pos_tags, self._get_column(grid, self._colmap['chunk'])))