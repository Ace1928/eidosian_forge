import copy
import numbers
from math import cos, sin, pi
import numpy as np
import ase.units as units
from ase.atom import Atom
from ase.cell import Cell
from ase.stress import voigt_6_to_full_3x3_stress, full_3x3_to_voigt_6_stress
from ase.data import atomic_masses, atomic_masses_common
from ase.geometry import (wrap_positions, find_mic, get_angles, get_distances,
from ase.symbols import Symbols, symbols2numbers
from ase.utils import deprecated
def _masked_rotate(self, center, axis, diff, mask):
    group = self.__class__()
    for i in range(len(self)):
        if mask[i]:
            group += self[i]
    group.translate(-center)
    group.rotate(diff * 180 / pi, axis)
    group.translate(center)
    j = 0
    for i in range(len(self)):
        if mask[i]:
            self.positions[i] = group[j].position
            j += 1