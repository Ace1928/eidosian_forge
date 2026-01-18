import numpy as np
import warnings
from ase.md.nvtberendsen import NVTBerendsen
import ase.units as units
class Inhomogeneous_NPTBerendsen(NPTBerendsen):
    """Berendsen (constant N, P, T) molecular dynamics.

    This dynamics scale the velocities and volumes to maintain a constant
    pressure and temperature.  The size of the unit cell is allowed to change
    independently in the three directions, but the angles remain constant.

    Usage: NPTBerendsen(atoms, timestep, temperature, taut, pressure, taup)

    atoms
        The list of atoms.

    timestep
        The time step.

    temperature
        The desired temperature, in Kelvin.

    taut
        Time constant for Berendsen temperature coupling.

    fixcm
        If True, the position and momentum of the center of mass is
        kept unperturbed.  Default: True.

    pressure
        The desired pressure, in bar (1 bar = 1e5 Pa).

    taup
        Time constant for Berendsen pressure coupling.

    compressibility
        The compressibility of the material, water 4.57E-5 bar-1, in bar-1

    mask
        Specifies which axes participate in the barostat.  Default (1, 1, 1)
        means that all axes participate, set any of them to zero to disable
        the barostat in that direction.
    """

    def __init__(self, atoms, timestep, temperature=None, *, temperature_K=None, taut=500.0 * units.fs, pressure=None, pressure_au=None, taup=1000.0 * units.fs, compressibility=None, compressibility_au=None, mask=(1, 1, 1), fixcm=True, trajectory=None, logfile=None, loginterval=1):
        NPTBerendsen.__init__(self, atoms, timestep, temperature=temperature, temperature_K=temperature_K, taut=taut, taup=taup, pressure=pressure, pressure_au=pressure_au, compressibility=compressibility, compressibility_au=compressibility_au, fixcm=fixcm, trajectory=trajectory, logfile=logfile, loginterval=loginterval)
        self.mask = mask

    def scale_positions_and_cell(self):
        """ Do the Berendsen pressure coupling,
        scale the atom position and the simulation cell."""
        taupscl = self.dt * self.compressibility / self.taup / 3.0
        stress = -self.atoms.get_stress(include_ideal_gas=True)
        if stress.shape == (6,):
            stress = stress[:3]
        elif stress.shape == (3, 3):
            stress = [stress[i][i] for i in range(3)]
        else:
            raise ValueError('Cannot use a stress tensor of shape ' + str(stress.shape))
        pbc = self.atoms.get_pbc()
        scl_pressurex = 1.0 - taupscl * (self.pressure - stress[0]) * pbc[0] * self.mask[0]
        scl_pressurey = 1.0 - taupscl * (self.pressure - stress[1]) * pbc[1] * self.mask[1]
        scl_pressurez = 1.0 - taupscl * (self.pressure - stress[2]) * pbc[2] * self.mask[2]
        cell = self.atoms.get_cell()
        cell = np.array([scl_pressurex * cell[0], scl_pressurey * cell[1], scl_pressurez * cell[2]])
        self.atoms.set_cell(cell, scale_atoms=True)