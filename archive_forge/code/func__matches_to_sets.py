import pytest
import networkx as nx
from networkx.algorithms import isomorphism as iso
def _matches_to_sets(matches):
    """
    Helper function to facilitate comparing collections of dictionaries in
    which order does not matter.
    """
    return {frozenset(m.items()) for m in matches}