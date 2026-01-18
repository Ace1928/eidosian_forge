from itertools import count
import numpy as np
from ase import Atoms
from ase.units import invcm, Ha
from ase.data import atomic_masses
from ase.calculators.calculator import all_changes
from ase.calculators.morse import MorsePotential
from ase.calculators.excitation_list import Excitation, ExcitationList
def H2Morse(state=0):
    """Return H2 as a Morse-Potential with calculator attached."""
    atoms = Atoms('H2', positions=np.zeros((2, 3)))
    atoms[1].position[2] = Re[state]
    atoms.calc = H2MorseCalculator(state=state)
    atoms.get_potential_energy()
    return atoms