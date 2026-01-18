import numpy as np
from math import cos, sin, pi
from ase.calculators.lammpslib import convert_cell
from ase.ga.utilities import (atoms_too_close,
from ase.ga.offspring_creator import OffspringCreator, CombinationMutation
from ase import Atoms
class PermuStrainMutation(CombinationMutation):
    """Combination of PermutationMutation and StrainMutation.

    For more information, see also:

      * `Lonie, Zurek, Comp. Phys. Comm. 182 (2011) 372-387`__

        __ https://doi.org/10.1016/j.cpc.2010.07.048

    Parameters:

    permutationmutation: OffspringCreator instance
        A mutation that permutes atom types.

    strainmutation: OffspringCreator instance
        A mutation that mutates by straining.
    """

    def __init__(self, permutationmutation, strainmutation, verbose=False):
        super(PermuStrainMutation, self).__init__(permutationmutation, strainmutation, verbose=verbose)
        self.descriptor = 'permustrain'