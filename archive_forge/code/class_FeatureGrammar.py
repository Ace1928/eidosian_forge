import re
from functools import total_ordering
from nltk.featstruct import SLASH, TYPE, FeatDict, FeatStruct, FeatStructReader
from nltk.internals import raise_unorderable_types
from nltk.probability import ImmutableProbabilisticMixIn
from nltk.util import invert_graph, transitive_closure
class FeatureGrammar(CFG):
    """
    A feature-based grammar.  This is equivalent to a
    ``CFG`` whose nonterminals are all
    ``FeatStructNonterminal``.

    A grammar consists of a start state and a set of
    productions.  The set of terminals and nonterminals
    is implicitly specified by the productions.
    """

    def __init__(self, start, productions):
        """
        Create a new feature-based grammar, from the given start
        state and set of ``Productions``.

        :param start: The start symbol
        :type start: FeatStructNonterminal
        :param productions: The list of productions that defines the grammar
        :type productions: list(Production)
        """
        CFG.__init__(self, start, productions)

    def _calculate_indexes(self):
        self._lhs_index = {}
        self._rhs_index = {}
        self._empty_index = {}
        self._empty_productions = []
        self._lexical_index = {}
        for prod in self._productions:
            lhs = self._get_type_if_possible(prod._lhs)
            if lhs not in self._lhs_index:
                self._lhs_index[lhs] = []
            self._lhs_index[lhs].append(prod)
            if prod._rhs:
                rhs0 = self._get_type_if_possible(prod._rhs[0])
                if rhs0 not in self._rhs_index:
                    self._rhs_index[rhs0] = []
                self._rhs_index[rhs0].append(prod)
            else:
                if lhs not in self._empty_index:
                    self._empty_index[lhs] = []
                self._empty_index[lhs].append(prod)
                self._empty_productions.append(prod)
            for token in prod._rhs:
                if is_terminal(token):
                    self._lexical_index.setdefault(token, set()).add(prod)

    @classmethod
    def fromstring(cls, input, features=None, logic_parser=None, fstruct_reader=None, encoding=None):
        """
        Return a feature structure based grammar.

        :param input: a grammar, either in the form of a string or else
        as a list of strings.
        :param features: a tuple of features (default: SLASH, TYPE)
        :param logic_parser: a parser for lambda-expressions,
        by default, ``LogicParser()``
        :param fstruct_reader: a feature structure parser
        (only if features and logic_parser is None)
        """
        if features is None:
            features = (SLASH, TYPE)
        if fstruct_reader is None:
            fstruct_reader = FeatStructReader(features, FeatStructNonterminal, logic_parser=logic_parser)
        elif logic_parser is not None:
            raise Exception("'logic_parser' and 'fstruct_reader' must not both be set")
        start, productions = read_grammar(input, fstruct_reader.read_partial, encoding=encoding)
        return cls(start, productions)

    def productions(self, lhs=None, rhs=None, empty=False):
        """
        Return the grammar productions, filtered by the left-hand side
        or the first item in the right-hand side.

        :param lhs: Only return productions with the given left-hand side.
        :param rhs: Only return productions with the given first item
            in the right-hand side.
        :param empty: Only return productions with an empty right-hand side.
        :rtype: list(Production)
        """
        if rhs and empty:
            raise ValueError('You cannot select empty and non-empty productions at the same time.')
        if not lhs and (not rhs):
            if empty:
                return self._empty_productions
            else:
                return self._productions
        elif lhs and (not rhs):
            if empty:
                return self._empty_index.get(self._get_type_if_possible(lhs), [])
            else:
                return self._lhs_index.get(self._get_type_if_possible(lhs), [])
        elif rhs and (not lhs):
            return self._rhs_index.get(self._get_type_if_possible(rhs), [])
        else:
            return [prod for prod in self._lhs_index.get(self._get_type_if_possible(lhs), []) if prod in self._rhs_index.get(self._get_type_if_possible(rhs), [])]

    def leftcorners(self, cat):
        """
        Return the set of all words that the given category can start with.
        Also called the "first set" in compiler construction.
        """
        raise NotImplementedError('Not implemented yet')

    def leftcorner_parents(self, cat):
        """
        Return the set of all categories for which the given category
        is a left corner.
        """
        raise NotImplementedError('Not implemented yet')

    def _get_type_if_possible(self, item):
        """
        Helper function which returns the ``TYPE`` feature of the ``item``,
        if it exists, otherwise it returns the ``item`` itself
        """
        if isinstance(item, dict) and TYPE in item:
            return FeatureValueType(item[TYPE])
        else:
            return item