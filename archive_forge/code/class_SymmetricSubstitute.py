import numpy as np
from operator import itemgetter
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.utilities import get_distance_matrix, get_nndist
from ase import Atoms
class SymmetricSubstitute(Mutation):
    """Permute all atoms within a subshell of the symmetric particle.
    The atoms within a subshell all have the same distance to the center,
    these are all equivalent under the particle point group symmetry.

    """

    def __init__(self, elements=None, num_muts=1, rng=np.random):
        Mutation.__init__(self, num_muts=num_muts, rng=rng)
        self.descriptor = 'SymmetricSubstitute'
        self.elements = elements

    def substitute(self, atoms):
        """Does the actual substitution"""
        atoms = atoms.copy()
        aconf = self.get_atomic_configuration(atoms, elements=self.elements)
        itbm = self.rng.randint(0, len(aconf) - 1)
        to_element = self.rng.choice(self.elements)
        for i in aconf[itbm]:
            atoms[i].symbol = to_element
        return atoms

    def get_new_individual(self, parents):
        f = parents[0]
        indi = self.substitute(f)
        indi = self.initialize_individual(f, indi)
        indi.info['data']['parents'] = [f.info['confid']]
        return (self.finalize_individual(indi), self.descriptor + ':Parent {0}'.format(f.info['confid']))