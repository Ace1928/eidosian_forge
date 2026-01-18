from ase.io import Trajectory
from ase.io import read
from ase.neb import NEB
from ase.optimize import BFGS
from ase.optimize import FIRE
from ase.calculators.singlepoint import SinglePointCalculator
import ase.parallel as mpi
import numpy as np
import shutil
import os
import types
from math import log
from math import exp
from contextlib import ExitStack
class seriel_writer:

    def __init__(self, traj, i, num):
        self.traj = traj
        self.i = i
        self.num = num

    def write(self):
        if self.num % (self.i + 1) == 0:
            self.traj.write()