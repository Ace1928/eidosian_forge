import textwrap
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tag import map_tag
from nltk.tree import Tree
from nltk.util import LazyConcatenation, LazyMap
def _get_parsed_sent(self, grid, pos_in_tree, tagset=None):
    words = self._get_column(grid, self._colmap['words'])
    pos_tags = self._get_column(grid, self._colmap['pos'])
    if tagset and tagset != self._tagset:
        pos_tags = [map_tag(self._tagset, tagset, t) for t in pos_tags]
    parse_tags = self._get_column(grid, self._colmap['tree'])
    treestr = ''
    for word, pos_tag, parse_tag in zip(words, pos_tags, parse_tags):
        if word == '(':
            word = '-LRB-'
        if word == ')':
            word = '-RRB-'
        if pos_tag == '(':
            pos_tag = '-LRB-'
        if pos_tag == ')':
            pos_tag = '-RRB-'
        left, right = parse_tag.split('*')
        right = right.count(')') * ')'
        treestr += f'{left} ({pos_tag} {word}) {right}'
    try:
        tree = self._tree_class.fromstring(treestr)
    except (ValueError, IndexError):
        tree = self._tree_class.fromstring(f'({self._root_label} {treestr})')
    if not pos_in_tree:
        for subtree in tree.subtrees():
            for i, child in enumerate(subtree):
                if isinstance(child, Tree) and len(child) == 1 and isinstance(child[0], str):
                    subtree[i] = (child[0], child.label())
    return tree