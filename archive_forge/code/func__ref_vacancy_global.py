import json
import numpy as np
import pytest
from ase.calculators.morse import MorsePotential
from ase.optimize import BFGS, ODE12r
from ase.optimize.precon import Exp
from ase.build import bulk
from ase.neb import NEB, NEBTools, NEBOptimizer
from ase.geometry.geometry import find_mic
from ase.constraints import FixBondLength
from ase.geometry.geometry import get_distances
from ase.utils.forcecurve import fit_images
@pytest.fixture(scope='module')
def _ref_vacancy_global(_setup_images_global):
    images, i1, i2 = _setup_images_global
    initial, saddle, final = (images[0].copy(), images[2].copy(), images[4].copy())
    initial.calc = calc()
    saddle.calc = calc()
    final.calc = calc()
    saddle.set_constraint(FixBondLength(i1, i2))
    opt = ODE12r(saddle)
    opt.run(fmax=0.01)
    nebtools = NEBTools([initial, saddle, final])
    Ef_ref, dE_ref = nebtools.get_barrier(fit=False)
    print('REF:', Ef_ref, dE_ref)
    return (Ef_ref, dE_ref, saddle)