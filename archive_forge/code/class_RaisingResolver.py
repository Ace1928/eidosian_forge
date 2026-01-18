from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
class RaisingResolver:
    """
    A partial fake L{IResolver} whose methods raise an exception containing the
    supplied arguments.
    """

    class QueryArguments(Exception):
        """
        Contains positional and keyword arguments in C{args}.
        """

    def query(self, *args, **kwargs):
        """
        Raises the supplied arguments.

        @param args: Positional arguments
        @type args: L{tuple}

        @param kwargs: Keyword args
        @type kwargs: L{dict}
        """
        raise self.QueryArguments(args, kwargs)