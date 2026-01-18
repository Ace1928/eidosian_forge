from typing import Callable
from zope.interface import Interface
class AlreadyQuit(Exception):
    """
    This worker worker is dead and cannot execute more instructions.
    """