import numpy as np
from itertools import combinations_with_replacement
from math import erf
from scipy.spatial.distance import cdist
from ase.neighborlist import NeighborList
from ase.utils import pbc2pbc
def _take_fingerprints(self, atoms, individual=False):
    """ Returns a [fingerprints,typedic] list, where fingerprints
        is a dictionary with the fingerprints, and typedic is a
        dictionary with the list of atom indices for each element
        (or "type") in the atoms object.
        The keys in the fingerprints dictionary are the (A,B) tuples,
        which are the different element-element combinations in the
        atoms object (A and B are the atomic numbers).
        When A != B, the (A,B) tuple is sorted (A < B).

        If individual=True, a dict is returned, where each atom index
        has an {atomic_number:fingerprint} dict as value.
        If individual=False, the fingerprints from atoms of the same
        atomic number are added together."""
    pos = atoms.get_positions()
    num = atoms.get_atomic_numbers()
    cell = atoms.get_cell()
    unique_types = np.unique(num)
    posdic = {}
    typedic = {}
    for t in unique_types:
        tlist = [i for i, atom in enumerate(atoms) if atom.number == t]
        typedic[t] = tlist
        posdic[t] = pos[tlist]
    volume, pmin, pmax, qmin, qmax = self._get_volume(atoms)
    non_pbc_dirs = [i for i in range(3) if not self.pbc[i]]

    def surface_area_0d(r):
        return 4 * np.pi * r ** 2

    def surface_area_1d(r, pos):
        q0 = pos[non_pbc_dirs[1]]
        phi1 = np.lib.scimath.arccos((qmax - q0) / r).real
        phi2 = np.pi - np.lib.scimath.arccos((qmin - q0) / r).real
        factor = 1 - (phi1 + phi2) / np.pi
        return surface_area_2d(r, pos) * factor

    def surface_area_2d(r, pos):
        p0 = pos[non_pbc_dirs[0]]
        area = np.minimum(pmax - p0, r) + np.minimum(p0 - pmin, r)
        area *= 2 * np.pi * r
        return area

    def surface_area_3d(r):
        return 4 * np.pi * r ** 2
    a = atoms.copy()
    a.set_pbc(self.pbc)
    nl = NeighborList([self.rcut / 2.0] * len(a), skin=0.0, self_interaction=False, bothways=True)
    nl.update(a)
    m = int(np.ceil(self.nsigma * self.sigma / self.binwidth))
    x = 0.25 * np.sqrt(2) * self.binwidth * (2 * m + 1) * 1.0 / self.sigma
    smearing_norm = erf(x)
    nbins = int(np.ceil(self.rcut * 1.0 / self.binwidth))
    bindist = self.binwidth * np.arange(1, nbins + 1)

    def take_individual_rdf(index, unique_type):
        rdf = np.zeros(nbins)
        if self.dimensions == 3:
            weights = 1.0 / surface_area_3d(bindist)
        elif self.dimensions == 2:
            weights = 1.0 / surface_area_2d(bindist, pos[index])
        elif self.dimensions == 1:
            weights = 1.0 / surface_area_1d(bindist, pos[index])
        elif self.dimensions == 0:
            weights = 1.0 / surface_area_0d(bindist)
        weights /= self.binwidth
        indices, offsets = nl.get_neighbors(index)
        valid = np.where(num[indices] == unique_type)
        p = pos[indices[valid]] + np.dot(offsets[valid], cell)
        r = cdist(p, [pos[index]])
        bins = np.floor(r / self.binwidth)
        for i in range(-m, m + 1):
            newbins = bins + i
            valid = np.where((newbins >= 0) & (newbins < nbins))
            valid_bins = newbins[valid].astype(int)
            values = weights[valid_bins]
            c = 0.25 * np.sqrt(2) * self.binwidth * 1.0 / self.sigma
            values *= 0.5 * erf(c * (2 * i + 1)) - 0.5 * erf(c * (2 * i - 1))
            values /= smearing_norm
            for j, valid_bin in enumerate(valid_bins):
                rdf[valid_bin] += values[j]
        rdf /= len(typedic[unique_type]) * 1.0 / volume
        return rdf
    fingerprints = {}
    if individual:
        for i in range(len(atoms)):
            fingerprints[i] = {}
            for unique_type in unique_types:
                fingerprint = take_individual_rdf(i, unique_type)
                if self.dimensions > 0:
                    fingerprint -= 1
                fingerprints[i][unique_type] = fingerprint
    else:
        for t1, t2 in combinations_with_replacement(unique_types, r=2):
            key = (t1, t2)
            fingerprint = np.zeros(nbins)
            for i in typedic[t1]:
                fingerprint += take_individual_rdf(i, t2)
            fingerprint /= len(typedic[t1])
            if self.dimensions > 0:
                fingerprint -= 1
            fingerprints[key] = fingerprint
    return [fingerprints, typedic]