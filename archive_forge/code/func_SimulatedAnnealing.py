from sys import version_info as _swig_python_version_info
import weakref
def SimulatedAnnealing(self, maximize, v, step, initial_temperature):
    """ Creates a Simulated Annealing monitor."""
    return _pywrapcp.Solver_SimulatedAnnealing(self, maximize, v, step, initial_temperature)