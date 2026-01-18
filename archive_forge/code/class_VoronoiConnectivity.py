from __future__ import annotations
import collections
import itertools
from math import acos, pi
from typing import TYPE_CHECKING
from warnings import warn
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi
from pymatgen.analysis.local_env import JmolNN, VoronoiNN
from pymatgen.core import Composition, Element, PeriodicSite, Species
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
class VoronoiConnectivity:
    """
    Computes the solid angles swept out by the shared face of the voronoi
    polyhedron between two sites.
    """

    def __init__(self, structure: Structure, cutoff=10):
        """
        Args:
            structure (Structure): Input structure
            cutoff (float) Cutoff distance.
        """
        self.cutoff = cutoff
        self.structure = structure
        recip_vec = np.array(self.structure.lattice.reciprocal_lattice.abc)
        cutoff_vec = np.ceil(cutoff * recip_vec / (2 * pi))
        offsets = np.mgrid[-cutoff_vec[0]:cutoff_vec[0] + 1, -cutoff_vec[1]:cutoff_vec[1] + 1, -cutoff_vec[2]:cutoff_vec[2] + 1].T
        self.offsets = np.reshape(offsets, (-1, 3))
        self.cart_offsets = self.structure.lattice.get_cartesian_coords(self.offsets)

    @property
    def connectivity_array(self):
        """
        Provides connectivity array.

        Returns:
            connectivity: An array of shape [atom_i, atom_j, image_j]. atom_i is
            the index of the atom in the input structure. Since the second
            atom can be outside of the unit cell, it must be described
            by both an atom index and an image index. Array data is the
            solid angle of polygon between atom_i and image_j of atom_j
        """
        cart_coords = np.array(self.structure.cart_coords)
        all_sites = cart_coords[:, None, :] + self.cart_offsets[None, :, :]
        vt = Voronoi(all_sites.reshape((-1, 3)))
        n_images = all_sites.shape[1]
        cs = (len(self.structure), len(self.structure), len(self.cart_offsets))
        connectivity = np.zeros(cs)
        vts = np.array(vt.vertices)
        for (ki, kj), v in vt.ridge_dict.items():
            atom_i = ki // n_images
            atom_j = kj // n_images
            image_i = ki % n_images
            image_j = kj % n_images
            if image_i != n_images // 2 and image_j != n_images // 2:
                continue
            if image_i == n_images // 2:
                val = solid_angle(vt.points[ki], vts[v])
                connectivity[atom_i, atom_j, image_j] = val
            if image_j == n_images // 2:
                val = solid_angle(vt.points[kj], vts[v])
                connectivity[atom_j, atom_i, image_i] = val
            if -10.101 in vts[v]:
                warn('Found connectivity with infinite vertex. Cutoff is too low, and results may be incorrect')
        return connectivity

    @property
    def max_connectivity(self):
        """
        Returns the 2d array [site_i, site_j] that represents the maximum connectivity of
        site i to any periodic image of site j.
        """
        return np.max(self.connectivity_array, axis=2)

    def get_connections(self):
        """
        Returns a list of site pairs that are Voronoi Neighbors, along
        with their real-space distances.
        """
        con = []
        max_conn = self.max_connectivity
        for ii in range(max_conn.shape[0]):
            for jj in range(max_conn.shape[1]):
                if max_conn[ii][jj] != 0:
                    dist = self.structure.get_distance(ii, jj)
                    con.append([ii, jj, dist])
        return con

    def get_sitej(self, site_index, image_index):
        """
        Assuming there is some value in the connectivity array at indices
        (1, 3, 12). site_i can be obtained directly from the input structure
        (structure[1]). site_j can be obtained by passing 3, 12 to this function.

        Args:
            site_index (int): index of the site (3 in the example)
            image_index (int): index of the image (12 in the example)
        """
        atoms_n_occu = self.structure[site_index].species
        lattice = self.structure.lattice
        coords = self.structure[site_index].frac_coords + self.offsets[image_index]
        return PeriodicSite(atoms_n_occu, coords, lattice)