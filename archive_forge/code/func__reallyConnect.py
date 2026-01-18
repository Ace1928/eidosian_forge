import random
from zope.interface import implementer
from twisted.internet import error, interfaces
from twisted.names import client, dns
from twisted.names.error import DNSNameError
from twisted.python.compat import nativeString
def _reallyConnect(self):
    if self.stopAfterDNS:
        self.stopAfterDNS = 0
        return
    self.host, self.port = self.pickServer()
    assert self.host is not None, 'Must have a host to connect to.'
    assert self.port is not None, 'Must have a port to connect to.'
    connectFunc = getattr(self.reactor, self.connectFuncName)
    self.connector = connectFunc(self.host, self.port, _SRVConnector_ClientFactoryWrapper(self, self.factory), *self.connectFuncArgs, **self.connectFuncKwArgs)