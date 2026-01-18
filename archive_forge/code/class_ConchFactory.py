from typing import Dict
from zope.interface import implementer
from twisted.conch import avatar, error as econch, interfaces as iconch
from twisted.conch.insults import insults
from twisted.conch.ssh import factory, session
from twisted.python import components
class ConchFactory(factory.SSHFactory):
    publicKeys: Dict[bytes, bytes] = {}
    privateKeys: Dict[bytes, bytes] = {}

    def __init__(self, portal):
        self.portal = portal