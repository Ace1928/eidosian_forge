from __future__ import annotations
import copy
import logging
from typing import TYPE_CHECKING
from monty.json import MSONable
from pymatgen.analysis.graphs import MoleculeGraph, MolGraphSplitError
from pymatgen.analysis.local_env import OpenBabelNN, metal_edge_extender
from pymatgen.io.babel import BabelMolAdaptor
def _open_all_rings(self) -> None:
    """
        Having already generated all unique fragments that did not require ring opening,
        now we want to also obtain fragments that do require opening. We achieve this by
        looping through all unique fragments and opening each bond present in any ring
        we find. We also temporarily add the principle molecule graph to self.unique_fragments
        so that its rings are opened as well.
        """
    alph_formula = self.mol_graph.molecule.composition.alphabetical_formula
    mol_key = f'{alph_formula} E{len(self.mol_graph.graph.edges())}'
    self.all_unique_frag_dict[mol_key] = [self.mol_graph]
    new_frag_keys: dict[str, list] = {'0': []}
    new_frag_key_dict = {}
    for key in self.all_unique_frag_dict:
        for fragment in self.all_unique_frag_dict[key]:
            ring_edges = fragment.find_rings()
            if ring_edges != []:
                for bond in ring_edges[0]:
                    new_fragment = open_ring(fragment, [bond], self.opt_steps)
                    alph_formula = new_fragment.molecule.composition.alphabetical_formula
                    frag_key = f'{alph_formula} E{len(new_fragment.graph.edges())}'
                    if frag_key not in self.all_unique_frag_dict:
                        if frag_key not in new_frag_keys['0']:
                            new_frag_keys['0'].append(copy.deepcopy(frag_key))
                            new_frag_key_dict[frag_key] = copy.deepcopy([new_fragment])
                        else:
                            found = False
                            for unique_fragment in new_frag_key_dict[frag_key]:
                                if unique_fragment.isomorphic_to(new_fragment):
                                    found = True
                                    break
                            if not found:
                                new_frag_key_dict[frag_key].append(copy.deepcopy(new_fragment))
                    else:
                        found = False
                        for unique_fragment in self.all_unique_frag_dict[frag_key]:
                            if unique_fragment.isomorphic_to(new_fragment):
                                found = True
                                break
                        if not found:
                            self.all_unique_frag_dict[frag_key].append(copy.deepcopy(new_fragment))
    for key, value in new_frag_key_dict.items():
        self.all_unique_frag_dict[key] = copy.deepcopy(value)
    idx = 0
    while len(new_frag_keys[str(idx)]) != 0:
        new_frag_key_dict = {}
        idx += 1
        new_frag_keys[str(idx)] = []
        for key in new_frag_keys[str(idx - 1)]:
            for fragment in self.all_unique_frag_dict[key]:
                ring_edges = fragment.find_rings()
                if ring_edges != []:
                    for bond in ring_edges[0]:
                        new_fragment = open_ring(fragment, [bond], self.opt_steps)
                        alph_formula = new_fragment.molecule.composition.alphabetical_formula
                        frag_key = f'{alph_formula} E{len(new_fragment.graph.edges())}'
                        if frag_key not in self.all_unique_frag_dict:
                            if frag_key not in new_frag_keys[str(idx)]:
                                new_frag_keys[str(idx)].append(copy.deepcopy(frag_key))
                                new_frag_key_dict[frag_key] = copy.deepcopy([new_fragment])
                            else:
                                found = False
                                for unique_fragment in new_frag_key_dict[frag_key]:
                                    if unique_fragment.isomorphic_to(new_fragment):
                                        found = True
                                        break
                                if not found:
                                    new_frag_key_dict[frag_key].append(copy.deepcopy(new_fragment))
                        else:
                            found = False
                            for unique_fragment in self.all_unique_frag_dict[frag_key]:
                                if unique_fragment.isomorphic_to(new_fragment):
                                    found = True
                                    break
                            if not found:
                                self.all_unique_frag_dict[frag_key].append(copy.deepcopy(new_fragment))
        for key, value in new_frag_key_dict.items():
            self.all_unique_frag_dict[key] = copy.deepcopy(value)
    self.all_unique_frag_dict.pop(mol_key)