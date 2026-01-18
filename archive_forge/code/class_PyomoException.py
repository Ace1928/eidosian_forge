import inspect
import textwrap
class PyomoException(Exception):
    """
    Exception class for other Pyomo exceptions to inherit from,
    allowing Pyomo exceptions to be caught in a general way
    (e.g., in other applications that use Pyomo).
    """
    pass