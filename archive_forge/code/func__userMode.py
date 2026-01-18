from time import ctime, time
from zope.interface import implementer
from twisted import copyright
from twisted.cred import credentials, error as ecred, portal
from twisted.internet import defer, protocol
from twisted.python import failure, log, reflect
from twisted.python.components import registerAdapter
from twisted.spread import pb
from twisted.words import ewords, iwords
from twisted.words.protocols import irc
def _userMode(self, user, modes=None):
    if modes:
        self.sendMessage(irc.ERR_UNKNOWNMODE, ':Unknown MODE flag.')
    elif user is self.avatar:
        self.sendMessage(irc.RPL_UMODEIS, '+')
    else:
        self.sendMessage(irc.ERR_USERSDONTMATCH, ":You can't look at someone else's modes.")