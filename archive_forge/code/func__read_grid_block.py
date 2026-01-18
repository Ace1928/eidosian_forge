import textwrap
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tag import map_tag
from nltk.tree import Tree
from nltk.util import LazyConcatenation, LazyMap
def _read_grid_block(self, stream):
    grids = []
    for block in read_blankline_block(stream):
        block = block.strip()
        if not block:
            continue
        grid = [line.split(self.sep) for line in block.split('\n')]
        if grid[0][self._colmap.get('words', 0)] == '-DOCSTART-':
            del grid[0]
        for row in grid:
            if len(row) != len(grid[0]):
                raise ValueError('Inconsistent number of columns:\n%s' % block)
        grids.append(grid)
    return grids