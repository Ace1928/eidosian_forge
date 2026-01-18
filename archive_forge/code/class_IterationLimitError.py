import inspect
import textwrap
class IterationLimitError(PyomoException, RuntimeError):
    """A subclass of :py:class:`RuntimeError`, raised by an iterative method
    when the iteration limit is reached.

    TODO: solvers currently do not raise this exception, but probably
    should (at least when non-normal termination conditions are mapped
    to exceptions)

    """