from os.path import abspath
from xml.dom.minidom import parseString
from unittest import skipIf
from zope.interface.verify import verifyObject
from twisted.internet import defer, reactor
from twisted.logger import globalLogPublisher
from twisted.python import failure, filepath
from twisted.spread import pb
from twisted.spread.banana import SIZE_LIMIT
from twisted.test import proto_helpers
from twisted.trial.unittest import TestCase
from twisted.web import client, distrib, resource, server, static
from twisted.web.http_headers import Headers
from twisted.web.test._util import _render
from twisted.web.test.requesthelper import DummyChannel, DummyRequest
def _setupDistribServer(self, child):
    """
        Set up a resource on a distrib site using L{ResourcePublisher}.

        @param child: The resource to publish using distrib.

        @return: A tuple consisting of the host and port on which to contact
            the created site.
        """
    distribRoot = resource.Resource()
    distribRoot.putChild(b'child', child)
    distribSite = server.Site(distribRoot)
    self.f1 = distribFactory = PBServerFactory(distrib.ResourcePublisher(distribSite))
    distribPort = reactor.listenTCP(0, distribFactory, interface='127.0.0.1')
    self.addCleanup(distribPort.stopListening)
    addr = distribPort.getHost()
    self.sub = mainRoot = distrib.ResourceSubscription(addr.host, addr.port)
    mainSite = server.Site(mainRoot)
    mainPort = reactor.listenTCP(0, mainSite, interface='127.0.0.1')
    self.addCleanup(mainPort.stopListening)
    mainAddr = mainPort.getHost()
    return (mainPort, mainAddr)