import os
from itertools import chain
import nltk
from nltk.internals import Counter
from nltk.sem import drt, linearlogic
from nltk.sem.logic import (
from nltk.tag import BigramTagger, RegexpTagger, TrigramTagger, UnigramTagger
def _lookup_semtype_option(self, semtype, node, depgraph):
    relationships = frozenset((depgraph.nodes[dep]['rel'].lower() for dep in chain.from_iterable(node['deps'].values()) if depgraph.nodes[dep]['rel'].lower() not in OPTIONAL_RELATIONSHIPS))
    try:
        lookup = semtype[relationships]
    except KeyError:
        best_match = frozenset()
        for relset_option in set(semtype) - {None}:
            if len(relset_option) > len(best_match) and relset_option < relationships:
                best_match = relset_option
        if not best_match:
            if None in semtype:
                best_match = None
            else:
                return None
        lookup = semtype[best_match]
    return lookup