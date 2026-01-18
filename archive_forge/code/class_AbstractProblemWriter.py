from pyomo.common import Factory
class AbstractProblemWriter(object):
    """Base class that can write optimization problems."""

    def __init__(self, problem_format):
        self.format = problem_format

    def __call__(self, model, filename, solver_capability, **kwds):
        raise TypeError('Method __call__ undefined in writer for format ' + str(self.format))

    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass