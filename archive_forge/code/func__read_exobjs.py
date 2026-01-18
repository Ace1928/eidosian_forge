import numpy as np
import ase.units as u
from ase.vibrations.raman import Raman, RamanPhonons
from ase.vibrations.resonant_raman import ResonantRaman
from ase.calculators.excitation_list import polarizability
def _read_exobjs(self, sign):
    return [disp.read_exobj() for disp in self._signed_disps(sign)]