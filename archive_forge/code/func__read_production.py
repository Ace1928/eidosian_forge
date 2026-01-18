import re
from functools import total_ordering
from nltk.featstruct import SLASH, TYPE, FeatDict, FeatStruct, FeatStructReader
from nltk.internals import raise_unorderable_types
from nltk.probability import ImmutableProbabilisticMixIn
from nltk.util import invert_graph, transitive_closure
def _read_production(line, nonterm_parser, probabilistic=False):
    """
    Parse a grammar rule, given as a string, and return
    a list of productions.
    """
    pos = 0
    lhs, pos = nonterm_parser(line, pos)
    m = _ARROW_RE.match(line, pos)
    if not m:
        raise ValueError('Expected an arrow')
    pos = m.end()
    probabilities = [0.0]
    rhsides = [[]]
    while pos < len(line):
        m = _PROBABILITY_RE.match(line, pos)
        if probabilistic and m:
            pos = m.end()
            probabilities[-1] = float(m.group(1)[1:-1])
            if probabilities[-1] > 1.0:
                raise ValueError('Production probability %f, should not be greater than 1.0' % (probabilities[-1],))
        elif line[pos] in '\'"':
            m = _TERMINAL_RE.match(line, pos)
            if not m:
                raise ValueError('Unterminated string')
            rhsides[-1].append(m.group(1)[1:-1])
            pos = m.end()
        elif line[pos] == '|':
            m = _DISJUNCTION_RE.match(line, pos)
            probabilities.append(0.0)
            rhsides.append([])
            pos = m.end()
        else:
            nonterm, pos = nonterm_parser(line, pos)
            rhsides[-1].append(nonterm)
    if probabilistic:
        return [ProbabilisticProduction(lhs, rhs, prob=probability) for rhs, probability in zip(rhsides, probabilities)]
    else:
        return [Production(lhs, rhs) for rhs in rhsides]