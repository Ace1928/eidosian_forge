import re
from nltk.corpus.reader.api import CorpusReader, SyntaxCorpusReader
from nltk.corpus.reader.util import (
from nltk.parse import DependencyGraph
class KNBCorpusReader(SyntaxCorpusReader):
    """
    This class implements:
      - ``__init__``, which specifies the location of the corpus
        and a method for detecting the sentence blocks in corpus files.
      - ``_read_block``, which reads a block from the input stream.
      - ``_word``, which takes a block and returns a list of list of words.
      - ``_tag``, which takes a block and returns a list of list of tagged
        words.
      - ``_parse``, which takes a block and returns a list of parsed
        sentences.

    The structure of tagged words:
      tagged_word = (word(str), tags(tuple))
      tags = (surface, reading, lemma, pos1, posid1, pos2, posid2, pos3, posid3, others ...)

    Usage example

    >>> from nltk.corpus.util import LazyCorpusLoader
    >>> knbc = LazyCorpusLoader(
    ...     'knbc/corpus1',
    ...     KNBCorpusReader,
    ...     r'.*/KN.*',
    ...     encoding='euc-jp',
    ... )

    >>> len(knbc.sents()[0])
    9

    """

    def __init__(self, root, fileids, encoding='utf8', morphs2str=_morphs2str_default):
        """
        Initialize KNBCorpusReader
        morphs2str is a function to convert morphlist to str for tree representation
        for _parse()
        """
        SyntaxCorpusReader.__init__(self, root, fileids, encoding)
        self.morphs2str = morphs2str

    def _read_block(self, stream):
        return read_blankline_block(stream)

    def _word(self, t):
        res = []
        for line in t.splitlines():
            if not re.match('EOS|\\*|\\#|\\+', line):
                cells = line.strip().split(' ')
                res.append(cells[0])
        return res

    def _tag(self, t, tagset=None):
        res = []
        for line in t.splitlines():
            if not re.match('EOS|\\*|\\#|\\+', line):
                cells = line.strip().split(' ')
                res.append((cells[0], ' '.join(cells[1:])))
        return res

    def _parse(self, t):
        dg = DependencyGraph()
        i = 0
        for line in t.splitlines():
            if line[0] in '*+':
                cells = line.strip().split(' ', 3)
                m = re.match('([\\-0-9]*)([ADIP])', cells[1])
                assert m is not None
                node = dg.nodes[i]
                node.update({'address': i, 'rel': m.group(2), 'word': []})
                dep_parent = int(m.group(1))
                if dep_parent == -1:
                    dg.root = node
                else:
                    dg.nodes[dep_parent]['deps'].append(i)
                i += 1
            elif line[0] != '#':
                cells = line.strip().split(' ')
                morph = (cells[0], ' '.join(cells[1:]))
                dg.nodes[i - 1]['word'].append(morph)
        if self.morphs2str:
            for node in dg.nodes.values():
                node['word'] = self.morphs2str(node['word'])
        return dg.tree()