from functools import reduce
from itertools import combinations, chain
from math import factorial
from operator import mul
import numpy as np
from ase.units import kg, C, _hbar, kB
from ase.vibrations import Vibrations
Return FC factors and corresponding frequencies up to given order.

        Parameters
        ----------
        temperature: float
          Temperature in K. Vibronic levels are occupied by a
          Boltzman distribution.
        forces: array
          Forces on atoms in the exited electronic state
        order: int
          number of quanta taken into account, default

        Returns
        --------
        FC: 3 entry list
          FC[0] = FC factors for 0-0 and +-1 vibrational quantum
          FC[1] = FC factors for +-2 vibrational quanta
          FC[2] = FC factors for combinations
        frequencies: 3 entry list
          frequencies[0] correspond to FC[0]
          frequencies[1] correspond to FC[1]
          frequencies[2] correspond to FC[2]
        