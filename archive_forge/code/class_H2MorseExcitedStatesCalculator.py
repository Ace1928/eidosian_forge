from itertools import count
import numpy as np
from ase import Atoms
from ase.units import invcm, Ha
from ase.data import atomic_masses
from ase.calculators.calculator import all_changes
from ase.calculators.morse import MorsePotential
from ase.calculators.excitation_list import Excitation, ExcitationList
class H2MorseExcitedStatesCalculator:
    """First singlet excited states of H2 from Morse potentials"""

    def __init__(self, nstates=3):
        """
        Parameters
        ----------
        nstates: int
          Numer of states to calculate 0 < nstates < 4, default 3
        """
        assert nstates > 0 and nstates < 4
        self.nstates = nstates

    def calculate(self, atoms):
        """Calculate excitation spectrum

        Parameters
        ----------
        atoms: Ase atoms object
        """
        mc = [0, 0.8, 0.7, 0.7]
        mr = [0, 1.0, 0.5, 0.5]
        cgs = atoms.calc
        r = atoms.get_distance(0, 1)
        E0 = cgs.get_potential_energy(atoms)
        exl = H2MorseExcitedStates()
        for i in range(1, self.nstates + 1):
            hvec = cgs.wfs[0] * cgs.wfs[i]
            energy = Ha * (0.5 - 1.0 / 8) - E0
            calc = H2MorseCalculator(state=i)
            calc.calculate(atoms)
            energy += calc.get_potential_energy()
            mur = hvec * (mc[i] + (r - Re[0]) * mr[i])
            muv = mur
            exl.append(H2Excitation(energy, i, mur, muv))
        return exl