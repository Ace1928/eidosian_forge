from math import sqrt
import numpy as np
class DivergenceBreak:
    """ Helper class which can be attached to a structure optimization,
        in order to terminate diverging calculations.

        Parameters:

        atoms: Atoms object being optimized
        dyn: The relaxation object being used
        N: The maximum number of recent steps to be included in the
           evaluation of the slope 
        Nmin: The minimal amount of steps required before evaluating
              the slope
    """

    def __init__(self, atoms, dyn, N=15, Nmin=5):
        self.atoms = atoms
        self.dyn = dyn
        self.N = N
        self.Nmin = 5
        self.energies = []

    def write(self):
        """ The method called by the optimizer in each step. """
        if len(self.energies) >= self.N:
            self.energies.pop(0)
        self.energies.append(self.atoms.get_potential_energy())
        if len(self.energies) > self.Nmin:
            x = np.array(range(len(self.energies)))
            y = np.array(self.energies)
            A = np.vstack([x, np.ones(len(x))]).T
            slope, intersect = np.linalg.lstsq(A, y)[0]
            if len(self.energies) >= self.N and slope > 0:
                self.dyn.converged = lambda x: True