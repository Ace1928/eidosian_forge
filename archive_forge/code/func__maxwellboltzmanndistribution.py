import numpy as np
from ase.parallel import world
from ase import units
from ase.md.md import process_temperature
def _maxwellboltzmanndistribution(masses, temp, communicator=None, rng=None):
    """Return a Maxwell-Boltzmann distribution with a given temperature.

    Paremeters:

    masses: float
        The atomic masses.

    temp: float
        The temperature in electron volt.

    communicator: MPI communicator (optional)
        Communicator used to distribute an identical distribution to 
        all tasks.  Set to 'serial' to disable communication (setting to None
        gives the default).  Default: ase.parallel.world

    rng: numpy RNG (optional)
        The random number generator.  Default: np.random

    Returns:

    A numpy array with Maxwell-Boltzmann distributed momenta.
    """
    if rng is None:
        rng = np.random
    if communicator is None:
        communicator = world
    xi = rng.standard_normal((len(masses), 3))
    if communicator != 'serial':
        communicator.broadcast(xi, 0)
    momenta = xi * np.sqrt(masses * temp)[:, np.newaxis]
    return momenta