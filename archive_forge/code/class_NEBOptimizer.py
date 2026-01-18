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
class NEBOptimizer(Optimizer):
    """
    This optimizer applies an adaptive ODE solver to a NEB

    Details of the adaptive ODE solver are described in paper IV
    """

    def __init__(self, neb, restart=None, logfile='-', trajectory=None, master=None, append_trajectory=False, method='ODE', alpha=0.01, verbose=0, rtol=0.1, C1=0.01, C2=2.0):
        super().__init__(atoms=neb, restart=restart, logfile=logfile, trajectory=trajectory, master=master, append_trajectory=append_trajectory, force_consistent=False)
        self.neb = neb
        method = method.lower()
        methods = ['ode', 'static', 'krylov']
        if method not in methods:
            raise ValueError(f'method must be one of {methods}')
        self.method = method
        self.alpha = alpha
        self.verbose = verbose
        self.rtol = rtol
        self.C1 = C1
        self.C2 = C2

    def force_function(self, X):
        positions = X.reshape((self.neb.nimages - 2) * self.neb.natoms, 3)
        self.neb.set_positions(positions)
        forces = self.neb.get_forces().reshape(-1)
        return forces

    def get_residual(self, F=None, X=None):
        return self.neb.get_residual()

    def log(self):
        fmax = self.get_residual()
        T = time.localtime()
        if self.logfile is not None:
            name = f'{self.__class__.__name__}[{self.method}]'
            if self.nsteps == 0:
                args = (' ' * len(name), 'Step', 'Time', 'fmax')
                msg = '%s  %4s %8s %12s\n' % args
                self.logfile.write(msg)
            args = (name, self.nsteps, T[3], T[4], T[5], fmax)
            msg = '%s:  %3d %02d:%02d:%02d %12.4f\n' % args
            self.logfile.write(msg)
            self.logfile.flush()

    def callback(self, X, F=None):
        self.log()
        self.call_observers()
        self.nsteps += 1

    def run_ode(self, fmax):
        try:
            ode12r(self.force_function, self.neb.get_positions().reshape(-1), fmax=fmax, rtol=self.rtol, C1=self.C1, C2=self.C2, steps=self.max_steps, verbose=self.verbose, callback=self.callback, residual=self.get_residual)
            return True
        except OptimizerConvergenceError:
            return False

    def run_static(self, fmax):
        X = self.neb.get_positions().reshape(-1)
        for step in range(self.max_steps):
            F = self.force_function(X)
            if self.neb.get_residual() <= fmax:
                return True
            X += self.alpha * F
            self.callback(X)
        return False

    def run(self, fmax=0.05, steps=None, method=None):
        """
        Optimize images to obtain the minimum energy path

        Parameters
        ----------
        fmax - desired force tolerance
        steps - maximum number of steps
        """
        if steps:
            self.max_steps = steps
        if method is None:
            method = self.method
        if method == 'ode':
            return self.run_ode(fmax)
        elif method == 'static':
            return self.run_static(fmax)
        else:
            raise ValueError(f'unknown method: {self.method}')