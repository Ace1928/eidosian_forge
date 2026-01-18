from __future__ import annotations
import numpy as np
from monty.dev import requires
from monty.serialization import loadfn
from scipy.interpolate import InterpolatedUnivariateSpline
from pymatgen.core import Lattice, Structure
from pymatgen.phonon.bandstructure import PhononBandStructure, PhononBandStructureSymmLine
from pymatgen.phonon.dos import CompletePhononDos, PhononDos
from pymatgen.phonon.gruneisen import GruneisenParameter, GruneisenPhononBandStructureSymmLine
from pymatgen.phonon.thermal_displacements import ThermalDisplacementMatrices
from pymatgen.symmetry.bandstructure import HighSymmKpath
def _extrapolate_grun(b, distance, gruneisenparameter, gruneisenband, i, pa):
    leftover_fraction = (pa['nqpoint'] - i - 1) / pa['nqpoint']
    if leftover_fraction < 0.1:
        diff = abs(b['gruneisen'] - gruneisenparameter[-1][len(gruneisenband)]) / abs(gruneisenparameter[-2][len(gruneisenband)] - gruneisenparameter[-1][len(gruneisenband)])
        if diff > 2:
            x = list(range(len(distance)))
            y = [i[len(gruneisenband)] for i in gruneisenparameter]
            y = y[-len(x):]
            extrapolator = InterpolatedUnivariateSpline(x, y, k=5)
            g_extrapolated = extrapolator(len(distance))
            gruen = float(g_extrapolated)
        else:
            gruen = b['gruneisen']
    else:
        gruen = b['gruneisen']
    return gruen