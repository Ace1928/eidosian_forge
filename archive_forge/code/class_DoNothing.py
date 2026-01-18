from twisted.internet.error import ConnectionDone
from twisted.internet.protocol import Protocol
from twisted.python.failure import Failure
from twisted.trial import unittest
class DoNothing:
    """
    Object with methods that do nothing.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, attr):
        return lambda *args, **kwargs: None