from time import perf_counter
from nltk.featstruct import TYPE, FeatStruct, find_variables, unify
from nltk.grammar import (
from nltk.parse.chart import (
from nltk.sem import logic
from nltk.tree import Tree
class FeatureTreeEdge(TreeEdge):
    """
    A specialized tree edge that allows shared variable bindings
    between nonterminals on the left-hand side and right-hand side.

    Each ``FeatureTreeEdge`` contains a set of ``bindings``, i.e., a
    dictionary mapping from variables to values.  If the edge is not
    complete, then these bindings are simply stored.  However, if the
    edge is complete, then the constructor applies these bindings to
    every nonterminal in the edge whose symbol implements the
    interface ``SubstituteBindingsI``.
    """

    def __init__(self, span, lhs, rhs, dot=0, bindings=None):
        """
        Construct a new edge.  If the edge is incomplete (i.e., if
        ``dot<len(rhs)``), then store the bindings as-is.  If the edge
        is complete (i.e., if ``dot==len(rhs)``), then apply the
        bindings to all nonterminals in ``lhs`` and ``rhs``, and then
        clear the bindings.  See ``TreeEdge`` for a description of
        the other arguments.
        """
        if bindings is None:
            bindings = {}
        if dot == len(rhs) and bindings:
            lhs = self._bind(lhs, bindings)
            rhs = [self._bind(elt, bindings) for elt in rhs]
            bindings = {}
        TreeEdge.__init__(self, span, lhs, rhs, dot)
        self._bindings = bindings
        self._comparison_key = (self._comparison_key, tuple(sorted(bindings.items())))

    @staticmethod
    def from_production(production, index):
        """
        :return: A new ``TreeEdge`` formed from the given production.
            The new edge's left-hand side and right-hand side will
            be taken from ``production``; its span will be
            ``(index,index)``; and its dot position will be ``0``.
        :rtype: TreeEdge
        """
        return FeatureTreeEdge(span=(index, index), lhs=production.lhs(), rhs=production.rhs(), dot=0)

    def move_dot_forward(self, new_end, bindings=None):
        """
        :return: A new ``FeatureTreeEdge`` formed from this edge.
            The new edge's dot position is increased by ``1``,
            and its end index will be replaced by ``new_end``.
        :rtype: FeatureTreeEdge
        :param new_end: The new end index.
        :type new_end: int
        :param bindings: Bindings for the new edge.
        :type bindings: dict
        """
        return FeatureTreeEdge(span=(self._span[0], new_end), lhs=self._lhs, rhs=self._rhs, dot=self._dot + 1, bindings=bindings)

    def _bind(self, nt, bindings):
        if not isinstance(nt, FeatStructNonterminal):
            return nt
        return nt.substitute_bindings(bindings)

    def next_with_bindings(self):
        return self._bind(self.nextsym(), self._bindings)

    def bindings(self):
        """
        Return a copy of this edge's bindings dictionary.
        """
        return self._bindings.copy()

    def variables(self):
        """
        :return: The set of variables used by this edge.
        :rtype: set(Variable)
        """
        return find_variables([self._lhs] + list(self._rhs) + list(self._bindings.keys()) + list(self._bindings.values()), fs_class=FeatStruct)

    def __str__(self):
        if self.is_complete():
            return super().__str__()
        else:
            bindings = '{%s}' % ', '.join(('%s: %r' % item for item in sorted(self._bindings.items())))
            return f'{super().__str__()} {bindings}'