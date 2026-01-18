import collections
import inspect
from automat import MethodicalMachine
from twisted.python.modules import PythonModule, getModule
class InvalidFQPN(Exception):
    """
    The given FQPN was not a dot-separated list of Python objects.
    """