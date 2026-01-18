import textwrap
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tag import map_tag
from nltk.tree import Tree
from nltk.util import LazyConcatenation, LazyMap
def _get_chunked_words(self, grid, chunk_types, tagset=None):
    words = self._get_column(grid, self._colmap['words'])
    pos_tags = self._get_column(grid, self._colmap['pos'])
    if tagset and tagset != self._tagset:
        pos_tags = [map_tag(self._tagset, tagset, t) for t in pos_tags]
    chunk_tags = self._get_column(grid, self._colmap['chunk'])
    stack = [Tree(self._root_label, [])]
    for word, pos_tag, chunk_tag in zip(words, pos_tags, chunk_tags):
        if chunk_tag == 'O':
            state, chunk_type = ('O', '')
        else:
            state, chunk_type = chunk_tag.split('-')
        if chunk_types is not None and chunk_type not in chunk_types:
            state = 'O'
        if state == 'I' and chunk_type != stack[-1].label():
            state = 'B'
        if state in 'BO' and len(stack) == 2:
            stack.pop()
        if state == 'B':
            new_chunk = Tree(chunk_type, [])
            stack[-1].append(new_chunk)
            stack.append(new_chunk)
        stack[-1].append((word, pos_tag))
    return stack[0]