from zope.interface import implementer
from zope.interface.verify import verifyClass
from twisted.internet.defer import Deferred, TimeoutError, gatherResults, succeed
from twisted.internet.interfaces import IResolverSimple
from twisted.names import client, root
from twisted.names.dns import (
from twisted.names.error import DNSNameError, ResolverError
from twisted.names.root import Resolver
from twisted.names.test.test_util import MemoryReactor
from twisted.python.log import msg
from twisted.trial import util
from twisted.trial.unittest import SynchronousTestCase, TestCase
class ResolverFactoryArguments(Exception):
    """
    Raised by L{raisingResolverFactory} with the *args and **kwargs passed to
    that function.
    """

    def __init__(self, args, kwargs):
        """
        Store the supplied args and kwargs as attributes.

        @param args: Positional arguments.
        @param kwargs: Keyword arguments.
        """
        self.args = args
        self.kwargs = kwargs