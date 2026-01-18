import re
from functools import total_ordering
from nltk.featstruct import SLASH, TYPE, FeatDict, FeatStruct, FeatStructReader
from nltk.internals import raise_unorderable_types
from nltk.probability import ImmutableProbabilisticMixIn
from nltk.util import invert_graph, transitive_closure
class PCFG(CFG):
    """
    A probabilistic context-free grammar.  A PCFG consists of a
    start state and a set of productions with probabilities.  The set of
    terminals and nonterminals is implicitly specified by the productions.

    PCFG productions use the ``ProbabilisticProduction`` class.
    ``PCFGs`` impose the constraint that the set of productions with
    any given left-hand-side must have probabilities that sum to 1
    (allowing for a small margin of error).

    If you need efficient key-based access to productions, you can use
    a subclass to implement it.

    :type EPSILON: float
    :cvar EPSILON: The acceptable margin of error for checking that
        productions with a given left-hand side have probabilities
        that sum to 1.
    """
    EPSILON = 0.01

    def __init__(self, start, productions, calculate_leftcorners=True):
        """
        Create a new context-free grammar, from the given start state
        and set of ``ProbabilisticProductions``.

        :param start: The start symbol
        :type start: Nonterminal
        :param productions: The list of productions that defines the grammar
        :type productions: list(Production)
        :raise ValueError: if the set of productions with any left-hand-side
            do not have probabilities that sum to a value within
            EPSILON of 1.
        :param calculate_leftcorners: False if we don't want to calculate the
            leftcorner relation. In that case, some optimized chart parsers won't work.
        :type calculate_leftcorners: bool
        """
        CFG.__init__(self, start, productions, calculate_leftcorners)
        probs = {}
        for production in productions:
            probs[production.lhs()] = probs.get(production.lhs(), 0) + production.prob()
        for lhs, p in probs.items():
            if not 1 - PCFG.EPSILON < p < 1 + PCFG.EPSILON:
                raise ValueError('Productions for %r do not sum to 1' % lhs)

    @classmethod
    def fromstring(cls, input, encoding=None):
        """
        Return a probabilistic context-free grammar corresponding to the
        input string(s).

        :param input: a grammar, either in the form of a string or else
             as a list of strings.
        """
        start, productions = read_grammar(input, standard_nonterm_parser, probabilistic=True, encoding=encoding)
        return cls(start, productions)