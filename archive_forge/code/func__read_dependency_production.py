import re
from functools import total_ordering
from nltk.featstruct import SLASH, TYPE, FeatDict, FeatStruct, FeatStructReader
from nltk.internals import raise_unorderable_types
from nltk.probability import ImmutableProbabilisticMixIn
from nltk.util import invert_graph, transitive_closure
def _read_dependency_production(s):
    if not _READ_DG_RE.match(s):
        raise ValueError('Bad production string')
    pieces = _SPLIT_DG_RE.split(s)
    pieces = [p for i, p in enumerate(pieces) if i % 2 == 1]
    lhside = pieces[0].strip('\'"')
    rhsides = [[]]
    for piece in pieces[2:]:
        if piece == '|':
            rhsides.append([])
        else:
            rhsides[-1].append(piece.strip('\'"'))
    return [DependencyProduction(lhside, rhside) for rhside in rhsides]