from __future__ import annotations
import collections
import copy
import math
import tempfile
from typing import TYPE_CHECKING, NamedTuple
import numpy as np
from monty.dev import deprecated
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import LightStructureEnvironments
from pymatgen.analysis.local_env import NearNeighbors
from pymatgen.electronic_structure.cohp import CompleteCohp
from pymatgen.electronic_structure.core import Spin
from pymatgen.electronic_structure.plotter import CohpPlotter
from pymatgen.io.lobster import Charge, Icohplist
from pymatgen.util.due import Doi, due
def _find_environments(self, additional_condition, lowerlimit, upperlimit, only_bonds_to):
    """
        Will find all relevant neighbors based on certain restrictions.

        Args:
            additional_condition (int): additional condition (see above)
            lowerlimit (float): lower limit that tells you which ICOHPs are considered
            upperlimit (float): upper limit that tells you which ICOHPs are considered
            only_bonds_to (list): list of str, e.g. ["O"] that will ensure that only bonds to "O" will be considered

        Returns:
            tuple: list of icohps, list of keys, list of lengths, list of neighisite, list of neighsite, list of coords
        """
    list_neighsite = []
    list_neighisite = []
    list_coords = []
    list_icohps = []
    list_lengths = []
    list_keys = []
    for idx in range(len(self.structure)):
        icohps = self._get_icohps(icohpcollection=self.Icohpcollection, isite=idx, lowerlimit=lowerlimit, upperlimit=upperlimit, only_bonds_to=only_bonds_to)
        additional_conds = self._find_relevant_atoms_additional_condition(idx, icohps, additional_condition)
        keys_from_ICOHPs, lengths_from_ICOHPs, neighbors_from_ICOHPs, selected_ICOHPs = additional_conds
        if len(neighbors_from_ICOHPs) > 0:
            centralsite = self.structure[idx]
            neighbors_by_distance_start = self.structure.get_sites_in_sphere(pt=centralsite.coords, r=np.max(lengths_from_ICOHPs) + 0.5, include_image=True, include_index=True)
            neighbors_by_distance = []
            list_distances = []
            index_here_list = []
            coords = []
            for neigh_new in sorted(neighbors_by_distance_start, key=lambda x: x[1]):
                site_here = neigh_new[0].to_unit_cell()
                index_here = neigh_new[2]
                index_here_list.append(index_here)
                cell_here = neigh_new[3]
                new_coords = [site_here.frac_coords[0] + float(cell_here[0]), site_here.frac_coords[1] + float(cell_here[1]), site_here.frac_coords[2] + float(cell_here[2])]
                coords.append(site_here.lattice.get_cartesian_coords(new_coords))
                neighbors_by_distance.append(neigh_new[0])
                list_distances.append(neigh_new[1])
            _list_neighsite = []
            _list_neighisite = []
            copied_neighbors_from_ICOHPs = copy.copy(neighbors_from_ICOHPs)
            copied_distances_from_ICOHPs = copy.copy(lengths_from_ICOHPs)
            _neigh_coords = []
            _neigh_frac_coords = []
            for ineigh, neigh in enumerate(neighbors_by_distance):
                index_here2 = index_here_list[ineigh]
                for idist, dist in enumerate(copied_distances_from_ICOHPs):
                    if np.isclose(dist, list_distances[ineigh], rtol=0.0001) and copied_neighbors_from_ICOHPs[idist] == index_here2:
                        _list_neighsite.append(neigh)
                        _list_neighisite.append(index_here2)
                        _neigh_coords.append(coords[ineigh])
                        _neigh_frac_coords.append(neigh.frac_coords)
                        del copied_distances_from_ICOHPs[idist]
                        del copied_neighbors_from_ICOHPs[idist]
                        break
            list_neighisite.append(_list_neighisite)
            list_neighsite.append(_list_neighsite)
            list_lengths.append(lengths_from_ICOHPs)
            list_keys.append(keys_from_ICOHPs)
            list_coords.append(_neigh_coords)
            list_icohps.append(selected_ICOHPs)
        else:
            list_neighsite.append([])
            list_neighisite.append([])
            list_icohps.append([])
            list_lengths.append([])
            list_keys.append([])
            list_coords.append([])
    return (list_icohps, list_keys, list_lengths, list_neighisite, list_neighsite, list_coords)