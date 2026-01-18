from __future__ import annotations
import copy
import logging
from typing import TYPE_CHECKING
from monty.json import MSONable
from pymatgen.analysis.graphs import MoleculeGraph, MolGraphSplitError
from pymatgen.analysis.local_env import OpenBabelNN, metal_edge_extender
from pymatgen.io.babel import BabelMolAdaptor
def _fragment_one_level(self, old_frag_dict: dict) -> dict:
    """
        Perform one step of iterative fragmentation on a list of molecule graphs. Loop through the graphs,
        then loop through each graph's edges and attempt to remove that edge in order to obtain two
        disconnected subgraphs, aka two new fragments. If successful, check to see if the new fragments
        are already present in self.unique_fragments, and append them if not. If unsuccessful, we know
        that edge belongs to a ring. If we are opening rings, do so with that bond, and then again
        check if the resulting fragment is present in self.unique_fragments and add it if it is not.
        """
    new_frag_dict = {}
    for old_frag_key in old_frag_dict:
        for old_frag in old_frag_dict[old_frag_key]:
            for edge in old_frag.graph.edges:
                bond = [(edge[0], edge[1])]
                fragments = []
                try:
                    fragments = old_frag.split_molecule_subgraphs(bond, allow_reverse=True)
                except MolGraphSplitError:
                    if self.open_rings:
                        fragments = [open_ring(old_frag, bond, self.opt_steps)]
                for fragment in fragments:
                    alph_formula = fragment.molecule.composition.alphabetical_formula
                    new_frag_key = f'{alph_formula} E{len(fragment.graph.edges())}'
                    proceed = True
                    if self.assume_previous_thoroughness and self.prev_unique_frag_dict != {} and (new_frag_key in self.prev_unique_frag_dict):
                        for unique_fragment in self.prev_unique_frag_dict[new_frag_key]:
                            if unique_fragment.isomorphic_to(fragment):
                                proceed = False
                                break
                    if proceed:
                        if new_frag_key not in self.all_unique_frag_dict:
                            self.all_unique_frag_dict[new_frag_key] = [fragment]
                            new_frag_dict[new_frag_key] = [fragment]
                        else:
                            found = False
                            for unique_fragment in self.all_unique_frag_dict[new_frag_key]:
                                if unique_fragment.isomorphic_to(fragment):
                                    found = True
                                    break
                            if not found:
                                self.all_unique_frag_dict[new_frag_key].append(fragment)
                                if new_frag_key in new_frag_dict:
                                    new_frag_dict[new_frag_key].append(fragment)
                                else:
                                    new_frag_dict[new_frag_key] = [fragment]
    return new_frag_dict