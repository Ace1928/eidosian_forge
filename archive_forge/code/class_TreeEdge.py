import itertools
import re
import warnings
from functools import total_ordering
from nltk.grammar import PCFG, is_nonterminal, is_terminal
from nltk.internals import raise_unorderable_types
from nltk.parse.api import ParserI
from nltk.tree import Tree
from nltk.util import OrderedDict
class TreeEdge(EdgeI):
    """
    An edge that records the fact that a tree is (partially)
    consistent with the sentence.  A tree edge consists of:

    - A span, indicating what part of the sentence is
      consistent with the hypothesized tree.
    - A left-hand side, specifying the hypothesized tree's node
      value.
    - A right-hand side, specifying the hypothesized tree's
      children.  Each element of the right-hand side is either a
      terminal, specifying a token with that terminal as its leaf
      value; or a nonterminal, specifying a subtree with that
      nonterminal's symbol as its node value.
    - A dot position, indicating which children are consistent
      with part of the sentence.  In particular, if ``dot`` is the
      dot position, ``rhs`` is the right-hand size, ``(start,end)``
      is the span, and ``sentence`` is the list of tokens in the
      sentence, then ``tokens[start:end]`` can be spanned by the
      children specified by ``rhs[:dot]``.

    For more information about edges, see the ``EdgeI`` interface.
    """

    def __init__(self, span, lhs, rhs, dot=0):
        """
        Construct a new ``TreeEdge``.

        :type span: tuple(int, int)
        :param span: A tuple ``(s, e)``, where ``tokens[s:e]`` is the
            portion of the sentence that is consistent with the new
            edge's structure.
        :type lhs: Nonterminal
        :param lhs: The new edge's left-hand side, specifying the
            hypothesized tree's node value.
        :type rhs: list(Nonterminal and str)
        :param rhs: The new edge's right-hand side, specifying the
            hypothesized tree's children.
        :type dot: int
        :param dot: The position of the new edge's dot.  This position
            specifies what prefix of the production's right hand side
            is consistent with the text.  In particular, if
            ``sentence`` is the list of tokens in the sentence, then
            ``okens[span[0]:span[1]]`` can be spanned by the
            children specified by ``rhs[:dot]``.
        """
        self._span = span
        self._lhs = lhs
        rhs = tuple(rhs)
        self._rhs = rhs
        self._dot = dot
        self._comparison_key = (span, lhs, rhs, dot)

    @staticmethod
    def from_production(production, index):
        """
        Return a new ``TreeEdge`` formed from the given production.
        The new edge's left-hand side and right-hand side will
        be taken from ``production``; its span will be
        ``(index,index)``; and its dot position will be ``0``.

        :rtype: TreeEdge
        """
        return TreeEdge(span=(index, index), lhs=production.lhs(), rhs=production.rhs(), dot=0)

    def move_dot_forward(self, new_end):
        """
        Return a new ``TreeEdge`` formed from this edge.
        The new edge's dot position is increased by ``1``,
        and its end index will be replaced by ``new_end``.

        :param new_end: The new end index.
        :type new_end: int
        :rtype: TreeEdge
        """
        return TreeEdge(span=(self._span[0], new_end), lhs=self._lhs, rhs=self._rhs, dot=self._dot + 1)

    def lhs(self):
        return self._lhs

    def span(self):
        return self._span

    def start(self):
        return self._span[0]

    def end(self):
        return self._span[1]

    def length(self):
        return self._span[1] - self._span[0]

    def rhs(self):
        return self._rhs

    def dot(self):
        return self._dot

    def is_complete(self):
        return self._dot == len(self._rhs)

    def is_incomplete(self):
        return self._dot != len(self._rhs)

    def nextsym(self):
        if self._dot >= len(self._rhs):
            return None
        else:
            return self._rhs[self._dot]

    def __str__(self):
        str = f'[{self._span[0]}:{self._span[1]}] '
        str += '%-2r ->' % (self._lhs,)
        for i in range(len(self._rhs)):
            if i == self._dot:
                str += ' *'
            str += ' %s' % repr(self._rhs[i])
        if len(self._rhs) == self._dot:
            str += ' *'
        return str

    def __repr__(self):
        return '[Edge: %s]' % self