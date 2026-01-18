import random
from zope.interface import implementer
from twisted.internet import error, interfaces
from twisted.names import client, dns
from twisted.names.error import DNSNameError
from twisted.python.compat import nativeString
class _SRVConnector_ClientFactoryWrapper:

    def __init__(self, connector, wrappedFactory):
        self.__connector = connector
        self.__wrappedFactory = wrappedFactory

    def startedConnecting(self, connector):
        self.__wrappedFactory.startedConnecting(self.__connector)

    def clientConnectionFailed(self, connector, reason):
        self.__connector.connectionFailed(reason)

    def clientConnectionLost(self, connector, reason):
        self.__connector.connectionLost(reason)

    def __getattr__(self, key):
        return getattr(self.__wrappedFactory, key)