import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
class CollectorClient(irc.IRCClient):
    """
    A client that saves in a list the names of the methods that got called.
    """

    def __init__(self, methodsList):
        """
        @param methodsList: list of methods' names that should be replaced.
        @type methodsList: C{list}
        """
        self.methods = []
        self.nickname = 'Wolf'
        for method in methodsList:

            def fake_method(method=method):
                """
                Collects C{method}s.
                """

                def inner(*args):
                    self.methods.append((method, args))
                return inner
            setattr(self, method, fake_method())