import random
import itertools
from ast import literal_eval
from Bio.Phylo import BaseTree
from Bio.Align import MultipleSeqAlignment
def _tree_to_bitstrs(tree):
    """Create a dict of a tree's clades to corresponding BitStrings (PRIVATE)."""
    clades_bitstrs = {}
    term_names = [term.name for term in tree.find_clades(terminal=True)]
    for clade in tree.find_clades(terminal=False):
        bitstr = _clade_to_bitstr(clade, term_names)
        clades_bitstrs[clade] = bitstr
    return clades_bitstrs