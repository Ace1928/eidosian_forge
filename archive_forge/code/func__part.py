import random
import itertools
from ast import literal_eval
from Bio.Phylo import BaseTree
from Bio.Align import MultipleSeqAlignment
def _part(clades):
    """Recursive function for Adam Consensus algorithm (PRIVATE)."""
    new_clade = None
    terms = clades[0].get_terminals()
    term_names = [term.name for term in terms]
    if len(terms) == 1 or len(terms) == 2:
        new_clade = clades[0]
    else:
        bitstrs = {_BitString('1' * len(terms))}
        for clade in clades:
            for child in clade.clades:
                bitstr = _clade_to_bitstr(child, term_names)
                to_remove = set()
                to_add = set()
                for bs in bitstrs:
                    if bs == bitstr:
                        continue
                    elif bs.contains(bitstr):
                        to_add.add(bitstr)
                        to_add.add(bs ^ bitstr)
                        to_remove.add(bs)
                    elif bitstr.contains(bs):
                        to_add.add(bs ^ bitstr)
                    elif not bs.independent(bitstr):
                        to_add.add(bs & bitstr)
                        to_add.add(bs & bitstr ^ bitstr)
                        to_add.add(bs & bitstr ^ bs)
                        to_remove.add(bs)
                bitstrs ^= to_remove
                if to_add:
                    for ta in sorted(to_add, key=lambda bs: bs.count('1')):
                        independent = True
                        for bs in bitstrs:
                            if not ta.independent(bs):
                                independent = False
                                break
                        if independent:
                            bitstrs.add(ta)
        new_clade = BaseTree.Clade()
        for bitstr in sorted(bitstrs):
            indices = bitstr.index_one()
            if len(indices) == 1:
                new_clade.clades.append(terms[indices[0]])
            elif len(indices) == 2:
                bifur_clade = BaseTree.Clade()
                bifur_clade.clades.append(terms[indices[0]])
                bifur_clade.clades.append(terms[indices[1]])
                new_clade.clades.append(bifur_clade)
            elif len(indices) > 2:
                part_names = [term_names[i] for i in indices]
                next_clades = []
                for clade in clades:
                    next_clades.append(_sub_clade(clade, part_names))
                new_clade.clades.append(_part(next_clades))
    return new_clade