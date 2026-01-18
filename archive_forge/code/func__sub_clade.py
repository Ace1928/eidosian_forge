import random
import itertools
from ast import literal_eval
from Bio.Phylo import BaseTree
from Bio.Align import MultipleSeqAlignment
def _sub_clade(clade, term_names):
    """Extract a compatible subclade that only contains the given terminal names (PRIVATE)."""
    term_clades = [clade.find_any(name) for name in term_names]
    sub_clade = clade.common_ancestor(term_clades)
    if len(term_names) != sub_clade.count_terminals():
        temp_clade = BaseTree.Clade()
        temp_clade.clades.extend(term_clades)
        for c in sub_clade.find_clades(terminal=False, order='preorder'):
            if c == sub_clade.root:
                continue
            childs = set(c.find_clades(terminal=True)) & set(term_clades)
            if childs:
                for tc in temp_clade.find_clades(terminal=False, order='preorder'):
                    tc_childs = set(tc.clades)
                    tc_new_clades = tc_childs - childs
                    if childs.issubset(tc_childs) and tc_new_clades:
                        tc.clades = list(tc_new_clades)
                        child_clade = BaseTree.Clade()
                        child_clade.clades.extend(list(childs))
                        tc.clades.append(child_clade)
        sub_clade = temp_clade
    return sub_clade