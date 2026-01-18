import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def _clientTestImpl(self, sender, group, type, msg, func, **kw):
    ident = pop(kw, 'ident', 'ident')
    host = pop(kw, 'host', 'host')
    wholeUser = sender + '!' + ident + '@' + host
    message = ':' + wholeUser + ' ' + type + ' ' + group + ' :' + msg + '\r\n'
    self.client.dataReceived(message)
    self.assertEqual(self.client.calls, [(func, kw)])
    self.client.calls = []