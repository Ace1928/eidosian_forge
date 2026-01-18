import numpy as np
from ase.parallel import world, broadcast
from ase.geometry import get_distances
Randomly attach two structures with a given minimal distance
      and ensure that these are distributed.

    Parameters
    ----------
    atoms1: Atoms object
    atoms2: Atoms object
    distance: float
      Required distance
    rng: random number generator object
      defaults to np.random.RandomState()
    comm: communicator to distribute
      Communicator to distribute the structure, default: world

    Returns
    -------
    Joined structure as an atoms object.
    