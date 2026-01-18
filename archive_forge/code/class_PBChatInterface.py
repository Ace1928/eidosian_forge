from zope.interface import provider
from twisted.application.service import ServiceMaker
from twisted.plugin import IPlugin
from twisted.words import iwords
@provider(IPlugin, iwords.IProtocolPlugin)
class PBChatInterface:
    name = 'pb'

    @classmethod
    def getFactory(cls, realm, portal):
        from twisted.spread import pb
        return pb.PBServerFactory(portal, True)