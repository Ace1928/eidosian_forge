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
def __initialize__(self):
    """Load files from the filesystem."""
    if not os.path.isfile('%s000.traj' % self.prefix):
        raise IOError('No file with name %s000.traj' % self.prefix, 'was found. Should contain initial image')
    index_exists = [i for i in range(self.n_max) if os.path.isfile('%s%03d.traj' % (self.prefix, i))]
    n_cur = index_exists[-1] + 1
    if self.world.rank == 0:
        print('The NEB initially has %d images ' % len(index_exists), '(including the end-points)')
    if len(index_exists) == 1:
        raise Exception('Only a start point exists')
    for i in range(len(index_exists)):
        if i != index_exists[i]:
            raise Exception('Files must be ordered sequentially', 'without gaps.')
    if self.world.rank == 0:
        for i in index_exists:
            filename_ref = self.iter_folder + '/%s%03diter000.traj' % (self.prefix, i)
            if os.path.isfile(filename_ref):
                try:
                    os.rename(filename_ref, filename_ref + '.bak')
                except IOError:
                    pass
            filename = '%s%03d.traj' % (self.prefix, i)
            try:
                shutil.copy2(filename, filename_ref)
            except IOError:
                pass
    self.world.barrier()
    for i in range(n_cur):
        if i in index_exists:
            filename = '%s%03d.traj' % (self.prefix, i)
            newim = read(filename)
            self.all_images.append(newim)
        else:
            self.all_images.append(self.all_images[0].copy())
    self.iteration = 0
    return n_cur