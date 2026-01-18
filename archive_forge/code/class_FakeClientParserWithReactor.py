from zope.interface.declarations import implementer
from twisted.internet.interfaces import (
from twisted.plugin import IPlugin
@implementer(IStreamClientEndpointStringParserWithReactor)
class FakeClientParserWithReactor(PluginBase):

    def parseStreamClient(self, *a, **kw):
        return StreamClient(self, a, kw)