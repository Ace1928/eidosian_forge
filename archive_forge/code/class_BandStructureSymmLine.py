from __future__ import annotations
import collections
import itertools
import math
import re
import warnings
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.json import MSONable
from pymatgen.core import Element, Lattice, Structure, get_el_sp
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.coord import pbc_diff
class BandStructureSymmLine(BandStructure, MSONable):
    """This object stores band structures along selected (symmetry) lines in the
    Brillouin zone. We call the different symmetry lines (ex: \\\\Gamma to Z)
    "branches".
    """

    def __init__(self, kpoints, eigenvals, lattice, efermi, labels_dict, coords_are_cartesian=False, structure=None, projections=None) -> None:
        """
        Args:
            kpoints: list of kpoint as numpy arrays, in frac_coords of the
                given lattice by default
            eigenvals: dict of energies for spin up and spin down
                {Spin.up:[][],Spin.down:[][]}, the first index of the array
                [][] refers to the band and the second to the index of the
                kpoint. The kpoints are ordered according to the order of the
                kpoints array. If the band structure is not spin polarized, we
                only store one data set under Spin.up.
            lattice: The reciprocal lattice.
                Pymatgen uses the physics convention of reciprocal lattice vectors
                WITH a 2*pi coefficient
            efermi: fermi energy
            labels_dict: (dict) of {} this link a kpoint (in frac coords or
                Cartesian coordinates depending on the coords).
            coords_are_cartesian: Whether coordinates are cartesian.
            structure: The crystal structure (as a pymatgen Structure object)
                associated with the band structure. This is needed if we
                provide projections to the band structure.
            projections: dict of orbital projections as {spin: array}. The
                indices of the array are [band_index, kpoint_index, orbital_index,
                ion_index].If the band structure is not spin polarized, we only
                store one data set under Spin.up.
        """
        super().__init__(kpoints, eigenvals, lattice, efermi, labels_dict, coords_are_cartesian, structure, projections)
        self.distance = []
        self.branches = []
        one_group: list = []
        branches_tmp = []
        previous_kpoint = self.kpoints[0]
        previous_distance = 0.0
        previous_label = self.kpoints[0].label
        for i, kpt in enumerate(self.kpoints):
            label = kpt.label
            if label is not None and previous_label is not None:
                self.distance.append(previous_distance)
            else:
                self.distance.append(np.linalg.norm(kpt.cart_coords - previous_kpoint.cart_coords) + previous_distance)
            previous_kpoint = kpt
            previous_distance = self.distance[i]
            if label and previous_label:
                if len(one_group) != 0:
                    branches_tmp.append(one_group)
                one_group = []
            previous_label = label
            one_group.append(i)
        if len(one_group) != 0:
            branches_tmp.append(one_group)
        for branch in branches_tmp:
            self.branches.append({'start_index': branch[0], 'end_index': branch[-1], 'name': f'{self.kpoints[branch[0]].label}-{self.kpoints[branch[-1]].label}'})
        self.is_spin_polarized = False
        if len(self.bands) == 2:
            self.is_spin_polarized = True

    def get_equivalent_kpoints(self, index):
        """Returns the list of kpoint indices equivalent (meaning they are the
        same frac coords) to the given one.

        Args:
            index: the kpoint index

        Returns:
            a list of equivalent indices

        TODO: now it uses the label we might want to use coordinates instead
        (in case there was a mislabel)
        """
        if self.kpoints[index].label is None:
            return [index]
        list_index_kpoints = []
        for i, kpt in enumerate(self.kpoints):
            if kpt.label == self.kpoints[index].label:
                list_index_kpoints.append(i)
        return list_index_kpoints

    def get_branch(self, index):
        """Returns in what branch(es) is the kpoint. There can be several
        branches.

        Args:
            index: the kpoint index

        Returns:
            A list of dictionaries [{"name","start_index","end_index","index"}]
            indicating all branches in which the k_point is. It takes into
            account the fact that one kpoint (e.g., \\\\Gamma) can be in several
            branches
        """
        to_return = []
        for idx in self.get_equivalent_kpoints(index):
            for b in self.branches:
                if b['start_index'] <= idx <= b['end_index']:
                    to_return.append({'name': b['name'], 'start_index': b['start_index'], 'end_index': b['end_index'], 'index': idx})
        return to_return

    def apply_scissor(self, new_band_gap):
        """Apply a scissor operator (shift of the CBM) to fit the given band gap.
        If it's a metal, we look for the band crossing the Fermi level
        and shift this one up. This will not work all the time for metals!

        Args:
            new_band_gap: the band gap the scissor band structure need to have.

        Returns:
            BandStructureSymmLine: with the applied scissor shift
        """
        if self.is_metal():
            max_index = -1000
            for idx in range(self.nb_bands):
                below = False
                above = False
                for j in range(len(self.kpoints)):
                    if self.bands[Spin.up][idx][j] < self.efermi:
                        below = True
                    if self.bands[Spin.up][idx][j] > self.efermi:
                        above = True
                if above and below and (idx > max_index):
                    max_index = idx
                if self.is_spin_polarized:
                    below = False
                    above = False
                    for j in range(len(self.kpoints)):
                        if self.bands[Spin.down][idx][j] < self.efermi:
                            below = True
                        if self.bands[Spin.down][idx][j] > self.efermi:
                            above = True
                    if above and below and (idx > max_index):
                        max_index = idx
            old_dict = self.as_dict()
            shift = new_band_gap
            for spin in old_dict['bands']:
                for k in range(len(old_dict['bands'][spin])):
                    for v in range(len(old_dict['bands'][spin][k])):
                        if k >= max_index:
                            old_dict['bands'][spin][k][v] = old_dict['bands'][spin][k][v] + shift
        else:
            shift = new_band_gap - self.get_band_gap()['energy']
            old_dict = self.as_dict()
            for spin in old_dict['bands']:
                for k in range(len(old_dict['bands'][spin])):
                    for v in range(len(old_dict['bands'][spin][k])):
                        if old_dict['bands'][spin][k][v] >= old_dict['cbm']['energy']:
                            old_dict['bands'][spin][k][v] = old_dict['bands'][spin][k][v] + shift
            old_dict['efermi'] = old_dict['efermi'] + shift
        return self.from_dict(old_dict)

    def as_dict(self):
        """JSON-serializable dict representation of BandStructureSymmLine."""
        dct = super().as_dict()
        dct['branches'] = self.branches
        return dct