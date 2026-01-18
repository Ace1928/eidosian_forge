import numpy as np
from math import cos, sin, pi
from ase.calculators.lammpslib import convert_cell
from ase.ga.utilities import (atoms_too_close,
from ase.ga.offspring_creator import OffspringCreator, CombinationMutation
from ase import Atoms
class StrainMutation(OffspringCreator):
    """ Mutates a candidate by applying a randomly generated strain.

    For more information, see also:

      * `Glass, Oganov, Hansen, Comp. Phys. Comm. 175 (2006) 713-720`__

        __ https://doi.org/10.1016/j.cpc.2006.07.020

      * `Lonie, Zurek, Comp. Phys. Comm. 182 (2011) 372-387`__

        __ https://doi.org/10.1016/j.cpc.2010.07.048

    After initialization of the mutation, a scaling volume
    (to which each mutated structure is scaled before checking the
    constraints) is typically generated from the population,
    which is then also occasionally updated in the course of the
    GA run.

    Parameters:

    blmin: dict
        The closest allowed interatomic distances on the form:
        {(Z, Z*): dist, ...}, where Z and Z* are atomic numbers.

    cellbounds: ase.ga.utilities.CellBounds instance
        Describes limits on the cell shape, see
        :class:`~ase.ga.utilities.CellBounds`.

    stddev: float
        Standard deviation used in the generation of the
        strain matrix elements.

    number_of_variable_cell_vectors: int (default 3)
        The number of variable cell vectors (1, 2 or 3).
        To keep things simple, it is the 'first' vectors which
        will be treated as variable, i.e. the 'a' vector in the
        univariate case, the 'a' and 'b' vectors in the bivariate
        case, etc.

    use_tags: boolean
        Whether to use the atomic tags to preserve molecular identity.

    rng: Random number generator
        By default numpy.random.
    """

    def __init__(self, blmin, cellbounds=None, stddev=0.7, number_of_variable_cell_vectors=3, use_tags=False, rng=np.random, verbose=False):
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.blmin = blmin
        self.cellbounds = cellbounds
        self.stddev = stddev
        self.number_of_variable_cell_vectors = number_of_variable_cell_vectors
        self.use_tags = use_tags
        self.scaling_volume = None
        self.descriptor = 'StrainMutation'
        self.min_inputs = 1

    def update_scaling_volume(self, population, w_adapt=0.5, n_adapt=0):
        """Function to initialize or update the scaling volume in a GA run.

        w_adapt: weight of the new vs the old scaling volume

        n_adapt: number of best candidates in the population that
                 are used to calculate the new scaling volume
        """
        if not n_adapt:
            n_adapt = int(np.ceil(0.2 * len(population)))
        v_new = np.mean([a.get_volume() for a in population[:n_adapt]])
        if not self.scaling_volume:
            self.scaling_volume = v_new
        else:
            volumes = [self.scaling_volume, v_new]
            weights = [1 - w_adapt, w_adapt]
            self.scaling_volume = np.average(volumes, weights=weights)

    def get_new_individual(self, parents):
        f = parents[0]
        indi = self.mutate(f)
        if indi is None:
            return (indi, 'mutation: strain')
        indi = self.initialize_individual(f, indi)
        indi.info['data']['parents'] = [f.info['confid']]
        return (self.finalize_individual(indi), 'mutation: strain')

    def mutate(self, atoms):
        """ Does the actual mutation. """
        cell_ref = atoms.get_cell()
        pos_ref = atoms.get_positions()
        vol_ref = atoms.get_volume()
        if self.use_tags:
            tags = atoms.get_tags()
            gather_atoms_by_tag(atoms)
            pos = atoms.get_positions()
        mutant = atoms.copy()
        count = 0
        too_close = True
        maxcount = 1000
        while too_close and count < maxcount:
            count += 1
            strain = np.identity(3)
            for i in range(self.number_of_variable_cell_vectors):
                for j in range(i + 1):
                    r = self.rng.normal(loc=0.0, scale=self.stddev)
                    if i == j:
                        strain[i, j] += r
                    else:
                        epsilon = 0.5 * r
                        strain[i, j] += epsilon
                        strain[j, i] += epsilon
            cell_new = np.dot(strain, cell_ref)
            cell_new = convert_cell(cell_new)[0].T
            if self.number_of_variable_cell_vectors > 0:
                volume = abs(np.linalg.det(cell_new))
                if self.scaling_volume is None:
                    scaling = vol_ref / volume
                else:
                    scaling = self.scaling_volume / volume
                scaling **= 1.0 / self.number_of_variable_cell_vectors
                cell_new[:self.number_of_variable_cell_vectors] *= scaling
            if not self.cellbounds.is_within_bounds(cell_new):
                continue
            for i in range(self.number_of_variable_cell_vectors, 3):
                assert np.allclose(cell_new[i], cell_ref[i])
            mutant.set_cell(cell_ref, scale_atoms=False)
            if self.use_tags:
                transfo = np.linalg.solve(cell_ref, cell_new)
                for tag in np.unique(tags):
                    select = np.where(tags == tag)
                    cop = np.mean(pos[select], axis=0)
                    disp = np.dot(cop, transfo) - cop
                    mutant.positions[select] += disp
            else:
                mutant.set_positions(pos_ref)
            mutant.set_cell(cell_new, scale_atoms=not self.use_tags)
            mutant.wrap()
            too_close = atoms_too_close(mutant, self.blmin, use_tags=self.use_tags)
        if count == maxcount:
            mutant = None
        return mutant