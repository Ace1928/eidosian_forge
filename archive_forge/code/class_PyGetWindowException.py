import sys, collections, pyrect
class PyGetWindowException(Exception):
    """
    Base class for exceptions raised when PyGetWindow functions
    encounter a problem. If PyGetWindow raises an exception that isn't
    this class, that indicates a bug in the module.
    """
    pass