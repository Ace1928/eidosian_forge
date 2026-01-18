from zope.interface.verify import verifyClass
from twisted.internet import defer
from twisted.internet.interfaces import IProtocolFactory
from twisted.names import dns, error, resolve, server
from twisted.python import failure, log
from twisted.trial import unittest
def _messageReceivedTest(self, methodName, message):
    """
        Assert that the named method is called with the given message when it is
        passed to L{DNSServerFactory.messageReceived}.

        @param methodName: The name of the method which is expected to be
            called.
        @type methodName: L{str}

        @param message: The message which is expected to be passed to the
            C{methodName} method.
        @type message: L{dns.Message}
        """
    message.queries = [None]
    receivedMessages = []

    def fakeHandler(message, protocol, address):
        receivedMessages.append((message, protocol, address))
    protocol = NoopProtocol()
    factory = server.DNSServerFactory(None)
    setattr(factory, methodName, fakeHandler)
    factory.messageReceived(message, protocol)
    self.assertEqual(receivedMessages, [(message, protocol, None)])