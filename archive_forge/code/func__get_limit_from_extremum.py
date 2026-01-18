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
def _get_limit_from_extremum(self, icohpcollection, percentage=0.15, adapt_extremum_to_add_cond=False, additional_condition=0):
    """
        Return limits for the evaluation of the icohp values from an icohpcollection
        Return -float("inf"), min(max_icohp*0.15,-0.1). Currently only works for ICOHPs.

        Args:
            icohpcollection: icohpcollection object
            percentage: will determine which ICOHPs or ICOOP or ICOBI will be considered
            (only 0.15 from the maximum value)
            adapt_extremum_to_add_cond: should the extrumum be adapted to the additional condition
            additional_condition: additional condition to determine which bonds are relevant

        Returns:
            tuple[float, float]: [-inf, min(strongest_icohp*0.15,-noise_cutoff)] / [max(strongest_icohp*0.15,
                noise_cutoff), inf]
        """
    if not adapt_extremum_to_add_cond or additional_condition == 0:
        extremum_based = icohpcollection.extremum_icohpvalue(summed_spin_channels=True) * percentage
    elif additional_condition == 1:
        list_icohps = []
        for value in icohpcollection._icohplist.values():
            atomnr1 = LobsterNeighbors._get_atomnumber(value._atom1)
            atomnr2 = LobsterNeighbors._get_atomnumber(value._atom2)
            val1 = self.valences[atomnr1]
            val2 = self.valences[atomnr2]
            if val1 < 0.0 < val2 or val2 < 0.0 < val1:
                list_icohps.append(value.summed_icohp)
        extremum_based = self._adapt_extremum_to_add_cond(list_icohps, percentage)
    elif additional_condition == 2:
        list_icohps = []
        for value in icohpcollection._icohplist.values():
            if value._atom1.rstrip('0123456789') != value._atom2.rstrip('0123456789'):
                list_icohps.append(value.summed_icohp)
        extremum_based = self._adapt_extremum_to_add_cond(list_icohps, percentage)
    elif additional_condition == 3:
        list_icohps = []
        for value in icohpcollection._icohplist.values():
            atomnr1 = LobsterNeighbors._get_atomnumber(value._atom1)
            atomnr2 = LobsterNeighbors._get_atomnumber(value._atom2)
            val1 = self.valences[atomnr1]
            val2 = self.valences[atomnr2]
            if (val1 < 0.0 < val2 or val2 < 0.0 < val1) and value._atom1.rstrip('0123456789') != value._atom2.rstrip('0123456789'):
                list_icohps.append(value.summed_icohp)
        extremum_based = self._adapt_extremum_to_add_cond(list_icohps, percentage)
    elif additional_condition == 4:
        list_icohps = []
        for value in icohpcollection._icohplist.values():
            if value._atom1.rstrip('0123456789') == 'O' or value._atom2.rstrip('0123456789') == 'O':
                list_icohps.append(value.summed_icohp)
        extremum_based = self._adapt_extremum_to_add_cond(list_icohps, percentage)
    elif additional_condition == 5:
        list_icohps = []
        for value in icohpcollection._icohplist.values():
            atomnr1 = LobsterNeighbors._get_atomnumber(value._atom1)
            atomnr2 = LobsterNeighbors._get_atomnumber(value._atom2)
            val1 = self.valences[atomnr1]
            val2 = self.valences[atomnr2]
            if val1 > 0.0 and val2 > 0.0 or (val1 < 0.0 and val2 < 0.0):
                list_icohps.append(value.summed_icohp)
        extremum_based = self._adapt_extremum_to_add_cond(list_icohps, percentage)
    elif additional_condition == 6:
        list_icohps = []
        for value in icohpcollection._icohplist.values():
            atomnr1 = LobsterNeighbors._get_atomnumber(value._atom1)
            atomnr2 = LobsterNeighbors._get_atomnumber(value._atom2)
            val1 = self.valences[atomnr1]
            val2 = self.valences[atomnr2]
            if val1 > 0.0 and val2 > 0.0:
                list_icohps.append(value.summed_icohp)
        extremum_based = self._adapt_extremum_to_add_cond(list_icohps, percentage)
    if not self.are_coops and (not self.are_cobis):
        max_here = min(extremum_based, -self.noise_cutoff) if self.noise_cutoff is not None else extremum_based
        return (-float('inf'), max_here)
    if self.are_coops or self.are_cobis:
        min_here = max(extremum_based, self.noise_cutoff) if self.noise_cutoff is not None else extremum_based
        return (min_here, float('inf'))
    return None