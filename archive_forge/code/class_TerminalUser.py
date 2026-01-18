from typing import Dict
from zope.interface import implementer
from twisted.conch import avatar, error as econch, interfaces as iconch
from twisted.conch.insults import insults
from twisted.conch.ssh import factory, session
from twisted.python import components
class TerminalUser(avatar.ConchUser, components.Adapter):

    def __init__(self, original, avatarId):
        components.Adapter.__init__(self, original)
        avatar.ConchUser.__init__(self)
        self.channelLookup[b'session'] = session.SSHSession