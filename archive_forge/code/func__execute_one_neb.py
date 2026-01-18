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
def _execute_one_neb(self, exitstack, n_cur, to_run, climb=False, many_steps=False):
    """Internal method which executes one NEB optimization."""
    closelater = exitstack.enter_context
    self.iteration += 1
    if self.world.rank == 0:
        for i in range(n_cur):
            if i not in to_run[1:-1]:
                filename = '%s%03d.traj' % (self.prefix, i)
                with Trajectory(filename, mode='w', atoms=self.all_images[i]) as traj:
                    traj.write()
                filename_ref = self.iter_folder + '/%s%03diter%03d.traj' % (self.prefix, i, self.iteration)
                if os.path.isfile(filename):
                    shutil.copy2(filename, filename_ref)
    if self.world.rank == 0:
        print('Now starting iteration %d on ' % self.iteration, to_run)
    self.attach_calculators([self.all_images[i] for i in to_run[1:-1]])
    neb = NEB([self.all_images[i] for i in to_run], k=[self.k[i] for i in to_run[0:-1]], method=self.method, parallel=self.parallel, remove_rotation_and_translation=self.remove_rotation_and_translation, climb=climb)
    qn = closelater(self.optimizer(neb, logfile=self.iter_folder + '/%s_log_iter%03d.log' % (self.prefix, self.iteration)))
    if self.parallel:
        nneb = to_run[0]
        nim = len(to_run) - 2
        n = self.world.size // nim
        j = 1 + self.world.rank // n
        assert nim * n == self.world.size
        traj = closelater(Trajectory('%s%03d.traj' % (self.prefix, j + nneb), 'w', self.all_images[j + nneb], master=self.world.rank % n == 0))
        filename_ref = self.iter_folder + '/%s%03diter%03d.traj' % (self.prefix, j + nneb, self.iteration)
        trajhist = closelater(Trajectory(filename_ref, 'w', self.all_images[j + nneb], master=self.world.rank % n == 0))
        qn.attach(traj)
        qn.attach(trajhist)
    else:
        num = 1
        for i, j in enumerate(to_run[1:-1]):
            filename_ref = self.iter_folder + '/%s%03diter%03d.traj' % (self.prefix, j, self.iteration)
            trajhist = closelater(Trajectory(filename_ref, 'w', self.all_images[j]))
            qn.attach(seriel_writer(trajhist, i, num).write)
            traj = closelater(Trajectory('%s%03d.traj' % (self.prefix, j), 'w', self.all_images[j]))
            qn.attach(seriel_writer(traj, i, num).write)
            num += 1
    if isinstance(self.maxsteps, (list, tuple)) and many_steps:
        steps = self.maxsteps[1]
    elif isinstance(self.maxsteps, (list, tuple)) and (not many_steps):
        steps = self.maxsteps[0]
    else:
        steps = self.maxsteps
    if isinstance(self.fmax, (list, tuple)) and many_steps:
        fmax = self.fmax[1]
    elif isinstance(self.fmax, (list, tuple)) and (not many_steps):
        fmax = self.fmax[0]
    else:
        fmax = self.fmax
    qn.run(fmax=fmax, steps=steps)
    neb.distribute = types.MethodType(store_E_and_F_in_spc, neb)
    neb.distribute()