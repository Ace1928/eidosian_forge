import numpy as np
from ase.parallel import world
from ase import units
from ase.md.md import process_temperature
def MaxwellBoltzmannDistribution(atoms, temp=None, *, temperature_K=None, communicator=None, force_temp=False, rng=None):
    """Sets the momenta to a Maxwell-Boltzmann distribution.

    Parameters:

    atoms: Atoms object
        The atoms.  Their momenta will be modified.

    temp: float (deprecated)
        The temperature in eV.  Deprecated, used temperature_K instead.

    temperature_K: float
        The temperature in Kelvin.

    communicator: MPI communicator (optional)
        Communicator used to distribute an identical distribution to
        all tasks.  Set to 'serial' to disable communication.  Leave as None to
        get the default: ase.parallel.world

    force_temp: bool (optinal, default: False)
        If True, random the momenta are rescaled so the kinetic energy is 
        exactly 3/2 N k T.  This is a slight deviation from the correct
        Maxwell-Boltzmann distribution.

    rng: Numpy RNG (optional)
        Random number generator.  Default: numpy.random
    """
    temp = units.kB * process_temperature(temp, temperature_K, 'eV')
    masses = atoms.get_masses()
    momenta = _maxwellboltzmanndistribution(masses, temp, communicator, rng)
    atoms.set_momenta(momenta)
    if force_temp:
        force_temperature(atoms, temperature=temp, unit='eV')