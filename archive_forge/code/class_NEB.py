import sys
import threading
import warnings
from abc import ABC, abstractmethod
import time
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import cumtrapz
import ase.parallel
from ase.build import minimize_rotation_and_translation
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.optimize import MDMin
from ase.optimize.optimize import Optimizer
from ase.optimize.sciopt import OptimizerConvergenceError
from ase.geometry import find_mic
from ase.utils import lazyproperty, deprecated
from ase.utils.forcecurve import fit_images
from ase.optimize.precon import Precon, PreconImages
from ase.optimize.ode import ode12r
class NEB(DyNEB):

    def __init__(self, images, k=0.1, climb=False, parallel=False, remove_rotation_and_translation=False, world=None, method='aseneb', allow_shared_calculator=False, precon=None, **kwargs):
        """Nudged elastic band.

        Paper I:

            G. Henkelman and H. Jonsson, Chem. Phys, 113, 9978 (2000).
            https://doi.org/10.1063/1.1323224

        Paper II:

            G. Henkelman, B. P. Uberuaga, and H. Jonsson, Chem. Phys,
            113, 9901 (2000).
            https://doi.org/10.1063/1.1329672

        Paper III:

            E. L. Kolsbjerg, M. N. Groves, and B. Hammer, J. Chem. Phys,
            145, 094107 (2016)
            https://doi.org/10.1063/1.4961868

        Paper IV:

            S. Makri, C. Ortner and J. R. Kermode, J. Chem. Phys.
            150, 094109 (2019)
            https://dx.doi.org/10.1063/1.5064465

        images: list of Atoms objects
            Images defining path from initial to final state.
        k: float or list of floats
            Spring constant(s) in eV/Ang.  One number or one for each spring.
        climb: bool
            Use a climbing image (default is no climbing image).
        parallel: bool
            Distribute images over processors.
        remove_rotation_and_translation: bool
            TRUE actives NEB-TR for removing translation and
            rotation during NEB. By default applied non-periodic
            systems
        method: string of method
            Choice betweeen five methods:

            * aseneb: standard ase NEB implementation
            * improvedtangent: Paper I NEB implementation
            * eb: Paper III full spring force implementation
            * spline: Paper IV spline interpolation (supports precon)
            * string: Paper IV string method (supports precon)
        allow_shared_calculator: bool
            Allow images to share the same calculator between them.
            Incompatible with parallelisation over images.
        precon: string, :class:`ase.optimize.precon.Precon` instance or list of
            instances. If present, enable preconditioing as in Paper IV. This is
            possible using the 'spline' or 'string' methods only.
            Default is no preconditioning (precon=None), which is converted to
            a list of :class:`ase.precon.precon.IdentityPrecon` instances.
        """
        for keyword in ('dynamic_relaxation', 'fmax', 'scale_fmax'):
            _check_deprecation(keyword, kwargs)
        defaults = dict(dynamic_relaxation=False, fmax=0.05, scale_fmax=0.0)
        defaults.update(kwargs)
        super().__init__(images, k=k, climb=climb, parallel=parallel, remove_rotation_and_translation=remove_rotation_and_translation, world=world, method=method, allow_shared_calculator=allow_shared_calculator, precon=precon, **defaults)