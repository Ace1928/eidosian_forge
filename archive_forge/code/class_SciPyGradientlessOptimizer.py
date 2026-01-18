import numpy as np
import scipy.optimize as opt
from ase.optimize.optimize import Optimizer
class SciPyGradientlessOptimizer(Optimizer):
    """General interface for gradient less SciPy optimizers

    Only the call to the optimizer is still needed

    Note: If you redefine x0() and f(), you don't even need an atoms object.
    Redefining these also allows you to specify an arbitrary objective
    function.

    XXX: This is still a work in progress
    """

    def __init__(self, atoms, logfile='-', trajectory=None, callback_always=False, master=None, force_consistent=None):
        """Initialize object

        Parameters:

        atoms: Atoms object
            The Atoms object to relax.

        trajectory: string
            Pickle file used to store trajectory of atomic movement.

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        callback_always: book
            Should the callback be run after each force call (also in the
            linesearch)

        alpha: float
            Initial guess for the Hessian (curvature of energy surface). A
            conservative value of 70.0 is the default, but number of needed
            steps to converge might be less if a lower value is used. However,
            a lower value also means risk of instability.

        master: boolean
            Defaults to None, which causes only rank 0 to save files.  If
            set to true,  this rank will save files.

        force_consistent: boolean or None
            Use force-consistent energy calls (as opposed to the energy
            extrapolated to 0 K).  By default (force_consistent=None) uses
            force-consistent energies if available in the calculator, but
            falls back to force_consistent=False if not.
        """
        restart = None
        Optimizer.__init__(self, atoms, restart, logfile, trajectory, master, force_consistent=force_consistent)
        self.function_calls = 0
        self.callback_always = callback_always

    def x0(self):
        """Return x0 in a way SciPy can use

        This class is mostly usable for subclasses wanting to redefine the
        parameters (and the objective function)"""
        return self.atoms.get_positions().reshape(-1)

    def f(self, x):
        """Objective function for use of the optimizers"""
        self.atoms.set_positions(x.reshape(-1, 3))
        self.function_calls += 1
        return self.atoms.get_potential_energy(force_consistent=self.force_consistent)

    def callback(self, x):
        """Callback function to be run after each iteration by SciPy

        This should also be called once before optimization starts, as SciPy
        optimizers only calls it after each iteration, while ase optimizers
        call something similar before as well.
        """
        self.call_observers()
        self.nsteps += 1

    def run(self, ftol=0.01, xtol=0.01, steps=100000000):
        if self.force_consistent is None:
            self.set_force_consistent()
        self.xtol = xtol
        self.ftol = ftol
        self.callback(None)
        try:
            self.call_fmin(xtol, ftol, steps)
        except Converged:
            pass

    def dump(self, data):
        pass

    def load(self):
        pass

    def call_fmin(self, fmax, steps):
        raise NotImplementedError