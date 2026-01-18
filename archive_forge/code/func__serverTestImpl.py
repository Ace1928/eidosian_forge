import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def _serverTestImpl(self, code, msg, func, **kw):
    host = pop(kw, 'host', 'server.host')
    nick = pop(kw, 'nick', 'nickname')
    args = pop(kw, 'args', '')
    message = ':' + host + ' ' + code + ' ' + nick + ' ' + args + ' :' + msg + '\r\n'
    self.client.dataReceived(message)
    self.assertEqual(self.client.calls, [(func, kw)])