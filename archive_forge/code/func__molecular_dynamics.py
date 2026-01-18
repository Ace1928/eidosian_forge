import os
import numpy as np
from ase import io, units
from ase.optimize import QuasiNewton
from ase.parallel import paropen, world
from ase.md import VelocityVerlet
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
def _molecular_dynamics(self, resume=None):
    """Performs a molecular dynamics simulation, until mdmin is
        exceeded. If resuming, the file number (md%05i) is expected."""
    self._log('msg', 'Molecular dynamics: md%05i' % self._counter)
    mincount = 0
    energies, oldpositions = ([], [])
    thermalized = False
    if resume:
        self._log('msg', 'Resuming MD from md%05i.traj' % resume)
        if os.path.getsize('md%05i.traj' % resume) == 0:
            self._log('msg', 'md%05i.traj is empty. Resuming from qn%05i.traj.' % (resume, resume - 1))
            atoms = io.read('qn%05i.traj' % (resume - 1), index=-1)
        else:
            with io.Trajectory('md%05i.traj' % resume, 'r') as images:
                for atoms in images:
                    energies.append(atoms.get_potential_energy())
                    oldpositions.append(atoms.positions.copy())
                    passedmin = self._passedminimum(energies)
                    if passedmin:
                        mincount += 1
            self._atoms.set_momenta(atoms.get_momenta())
            thermalized = True
        self._atoms.positions = atoms.get_positions()
        self._log('msg', 'Starting MD with %i existing energies.' % len(energies))
    if not thermalized:
        MaxwellBoltzmannDistribution(self._atoms, temperature_K=self._temperature, force_temp=True)
    traj = io.Trajectory('md%05i.traj' % self._counter, 'a', self._atoms)
    dyn = VelocityVerlet(self._atoms, timestep=self._timestep * units.fs)
    log = MDLogger(dyn, self._atoms, 'md%05i.log' % self._counter, header=True, stress=False, peratom=False)
    with traj, dyn, log:
        dyn.attach(log, interval=1)
        dyn.attach(traj, interval=1)
        while mincount < self._mdmin:
            dyn.run(1)
            energies.append(self._atoms.get_potential_energy())
            passedmin = self._passedminimum(energies)
            if passedmin:
                mincount += 1
            oldpositions.append(self._atoms.positions.copy())
        self._atoms.positions = oldpositions[passedmin[0]]