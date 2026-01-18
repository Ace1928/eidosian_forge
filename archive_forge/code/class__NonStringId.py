import unittest
from sys import exc_info
from twisted.python.failure import Failure
class _NonStringId:
    """
    A class that looks a little like a TestCase, but not enough so to
    actually be used as one.  This helps L{BrokenRunInfrastructure} use some
    interfaces incorrectly to provoke certain failure conditions.
    """

    def id(self) -> object:
        return object()