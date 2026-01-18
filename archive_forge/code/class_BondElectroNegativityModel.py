import inspect
import json
import numpy as np
from ase.data import covalent_radii
from ase.neighborlist import NeighborList
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.utilities import atoms_too_close, gather_atoms_by_tag
from scipy.spatial.distance import cdist
class BondElectroNegativityModel(PairwiseHarmonicPotential):
    """Pairwise harmonic potential where the force constants are
    determined using the "bond electronegativity" model, see:

    * `Lyakhov, Oganov, Valle, Comp. Phys. Comm. 181 (2010) 1623-1632`__

      __ https://dx.doi.org/10.1016/j.cpc.2010.06.007

    * `Lyakhov, Oganov, Phys. Rev. B 84 (2011) 092103`__

      __ https://dx.doi.org/10.1103/PhysRevB.84.092103
    """

    def calculate_force_constants(self):
        cell = self.atoms.get_cell()
        pos = self.atoms.get_positions()
        num = self.atoms.get_atomic_numbers()
        nat = len(self.atoms)
        nl = self.nl
        s_norms = []
        valence_states = []
        r_cov = []
        for i in range(nat):
            indices, offsets = nl.get_neighbors(i)
            p = pos[indices] + np.dot(offsets, cell)
            r = cdist(p, [pos[i]])
            r_ci = covalent_radii[num[i]]
            s = 0.0
            for j, index in enumerate(indices):
                d = r[j] - r_ci - covalent_radii[num[index]]
                s += np.exp(-d / 0.37)
            s_norms.append(s)
            valence_states.append(get_number_of_valence_electrons(num[i]))
            r_cov.append(r_ci)
        self.force_constants = []
        for i in range(nat):
            indices, offsets = nl.get_neighbors(i)
            p = pos[indices] + np.dot(offsets, cell)
            r = cdist(p, [pos[i]])[:, 0]
            fc = []
            for j, ii in enumerate(indices):
                d = r[j] - r_cov[i] - r_cov[ii]
                chi_ik = 0.481 * valence_states[i] / (r_cov[i] + 0.5 * d)
                chi_jk = 0.481 * valence_states[ii] / (r_cov[ii] + 0.5 * d)
                cn_ik = s_norms[i] / np.exp(-d / 0.37)
                cn_jk = s_norms[ii] / np.exp(-d / 0.37)
                fc.append(np.sqrt(chi_ik * chi_jk / (cn_ik * cn_jk)))
            self.force_constants.append(np.array(fc))