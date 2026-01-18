import sys
import time
import warnings
from math import cos, sin, atan, tan, degrees, pi, sqrt
from typing import Dict, Any
import numpy as np
from ase.optimize.optimize import Optimizer
from ase.parallel import world
from ase.calculators.singlepoint import SinglePointCalculator
from ase.utils import IOContext
class MinModeTranslate(Optimizer):
    """An Optimizer specifically tailored to minimum mode following."""

    def __init__(self, atoms, logfile='-', trajectory=None):
        Optimizer.__init__(self, atoms, None, logfile, trajectory)
        self.control = atoms.get_control()
        if self.logfile is not None:
            l = ''
            if isinstance(self.control, DimerControl):
                l = 'MinModeTranslate: STEP      TIME          ENERGY    ' + 'MAX-FORCE     STEPSIZE    CURVATURE  ROT-STEPS\n'
            self.logfile.write(l)
            self.logfile.flush()
        self.cg_on = self.control.get_parameter('cg_translation')
        self.trial_step = self.control.get_parameter('trial_trans_step')
        self.max_step = self.control.get_parameter('maximum_translation')
        if self.cg_on:
            self.cg_init = True

    def initialize(self):
        """Set initial values."""
        self.r0 = None
        self.f0 = None

    def step(self, f=None):
        """Perform the optimization step."""
        atoms = self.atoms
        if f is None:
            f = atoms.get_forces()
        r = atoms.get_positions()
        curv = atoms.get_curvature()
        f0p = f.copy()
        r0 = r.copy()
        direction = f0p.copy()
        if self.cg_on:
            direction = self.get_cg_direction(direction)
        direction = normalize(direction)
        if curv > 0.0:
            step = direction * self.max_step
        else:
            r0t = r0 + direction * self.trial_step
            f0tp = self.atoms.get_projected_forces(r0t)
            F = np.vdot(f0tp + f0p, direction) / 2.0
            C = np.vdot(f0tp - f0p, direction) / self.trial_step
            step = (-F / C + self.trial_step / 2.0) * direction
            if norm(step) > self.max_step:
                step = direction * self.max_step
        self.log(f0p, norm(step))
        atoms.set_positions(r + step)
        self.f0 = f.flat.copy()
        self.r0 = r.flat.copy()

    def get_cg_direction(self, direction):
        """Apply the Conjugate Gradient algorithm to the step direction."""
        if self.cg_init:
            self.cg_init = False
            self.direction_old = direction.copy()
            self.cg_direction = direction.copy()
        old_norm = np.vdot(self.direction_old, self.direction_old)
        if old_norm != 0.0:
            betaPR = np.vdot(direction, direction - self.direction_old) / old_norm
        else:
            betaPR = 0.0
        if betaPR < 0.0:
            betaPR = 0.0
        self.cg_direction = direction + self.cg_direction * betaPR
        self.direction_old = direction.copy()
        return self.cg_direction.copy()

    def log(self, f=None, stepsize=None):
        """Log each step of the optimization."""
        if f is None:
            f = self.atoms.get_forces()
        if self.logfile is not None:
            T = time.localtime()
            e = self.atoms.get_potential_energy()
            fmax = sqrt((f ** 2).sum(axis=1).max())
            rotsteps = self.atoms.control.get_counter('rotcount')
            curvature = self.atoms.get_curvature()
            l = ''
            if stepsize:
                if isinstance(self.control, DimerControl):
                    l = '%s: %4d  %02d:%02d:%02d %15.6f %12.4f %12.6f %12.6f %10d\n' % ('MinModeTranslate', self.nsteps, T[3], T[4], T[5], e, fmax, stepsize, curvature, rotsteps)
            elif isinstance(self.control, DimerControl):
                l = '%s: %4d  %02d:%02d:%02d %15.6f %12.4f %s %12.6f %10d\n' % ('MinModeTranslate', self.nsteps, T[3], T[4], T[5], e, fmax, '    --------', curvature, rotsteps)
            self.logfile.write(l)
            self.logfile.flush()