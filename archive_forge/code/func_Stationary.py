import numpy as np
from ase.parallel import world
from ase import units
from ase.md.md import process_temperature
def Stationary(atoms, preserve_temperature=True):
    """Sets the center-of-mass momentum to zero."""
    temp0 = atoms.get_temperature()
    p = atoms.get_momenta()
    p0 = np.sum(p, 0)
    m = atoms.get_masses()
    mtot = np.sum(m)
    v0 = p0 / mtot
    p -= v0 * m[:, np.newaxis]
    atoms.set_momenta(p)
    if preserve_temperature:
        force_temperature(atoms, temp0)