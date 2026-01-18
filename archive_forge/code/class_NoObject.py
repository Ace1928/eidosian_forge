import collections
import inspect
from automat import MethodicalMachine
from twisted.python.modules import PythonModule, getModule
class NoObject(InvalidFQPN):
    """
    A suffix of the FQPN was not an accessible object
    """