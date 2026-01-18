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
Determine which set of images to do a NEB at.
        The priority is to first include all images without valid energies,
        secondly include the highest energy image.