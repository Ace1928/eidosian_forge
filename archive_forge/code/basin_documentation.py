import numpy as np
from ase.optimize.optimize import Dynamics
from ase.optimize.fire import FIRE
from ase.units import kB
from ase.parallel import world
from ase.io.trajectory import Trajectory
Return the energy of the nearest local minimum.