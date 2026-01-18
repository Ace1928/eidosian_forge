from __future__ import annotations
import os
import warnings
import numpy as np
from monty.serialization import loadfn
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
class DLSVolumePredictor:
    """
    Data-mined lattice scaling (DLS) scheme that relies on data-mined bond
    lengths to predict the crystal volume of a given structure.

    As of 2/12/19, we suggest this method be used in conjunction with
    min_scaling and max_scaling to prevent instances of very large, unphysical
    predicted volumes found in a small subset of structures.
    """

    def __init__(self, cutoff=4.0, min_scaling=0.5, max_scaling=1.5):
        """
        Args:
            cutoff (float): cutoff radius added to site radius for finding
                site pairs. Necessary to increase only if your initial
                structure guess is extremely bad (atoms way too far apart). In
                all other instances, increasing cutoff gives same answer
                but takes more time.
            min_scaling (float): if not None, this will ensure that the new
                volume is at least this fraction of the original (preventing
                too-small volumes)
            max_scaling (float): if not None, this will ensure that the new
                volume is at most this fraction of the original (preventing
                too-large volumes).
        """
        self.cutoff = cutoff
        self.min_scaling = min_scaling
        self.max_scaling = max_scaling

    def predict(self, structure: Structure, icsd_vol=False):
        """
        Given a structure, returns the predicted volume.

        Args:
            structure (Structure) : a crystal structure with an unknown volume.
            icsd_vol (bool) : True if the input structure's volume comes from ICSD.

        Returns:
            a float value of the predicted volume.
        """
        std_x = np.std([site.specie.X for site in structure])
        sub_sites = []
        bp_dict = {}
        for sp in list(structure.composition):
            if sp.atomic_radius:
                sub_sites.extend([site for site in structure if site.specie == sp])
            else:
                warnings.warn(f'VolumePredictor: no atomic radius data for {sp}')
            if sp.symbol not in bond_params:
                warnings.warn(f'VolumePredictor: bond parameters not found, used atomic radii for {sp}')
            else:
                r, k = (bond_params[sp.symbol]['r'], bond_params[sp.symbol]['k'])
                bp_dict[sp] = float(r) + float(k) * std_x
        reduced_structure = Structure.from_sites(sub_sites)
        smallest_ratio = None
        for site1 in reduced_structure:
            sp1 = site1.specie
            neighbors = reduced_structure.get_neighbors(site1, sp1.atomic_radius + self.cutoff)
            for nn in neighbors:
                sp2 = nn.specie
                if sp1 in bp_dict and sp2 in bp_dict:
                    expected_dist = bp_dict[sp1] + bp_dict[sp2]
                else:
                    assert sp1.atomic_radius is not None
                    expected_dist = sp1.atomic_radius + sp2.atomic_radius
                if not smallest_ratio or nn.nn_distance / expected_dist < smallest_ratio:
                    smallest_ratio = nn.nn_distance / expected_dist
        if not smallest_ratio:
            raise ValueError('Could not find any bonds within the given cutoff in this structure.')
        volume_factor = (1 / smallest_ratio) ** 3
        if icsd_vol:
            volume_factor *= 1.05
        if self.min_scaling:
            volume_factor = max(self.min_scaling, volume_factor)
        if self.max_scaling:
            volume_factor = min(self.max_scaling, volume_factor)
        return structure.volume * volume_factor

    def get_predicted_structure(self, structure: Structure, icsd_vol=False):
        """
        Given a structure, returns back the structure scaled to predicted
        volume.

        Args:
            structure (Structure): structure w/unknown volume

        Returns:
            a Structure object with predicted volume
        """
        new_structure = structure.copy()
        new_structure.scale_lattice(self.predict(structure, icsd_vol=icsd_vol))
        return new_structure