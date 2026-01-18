import errno
import operator
import time
from twisted.internet import protocol, task
from twisted.internet.testing import StringIOWithoutClosing, StringTransport
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.words.protocols import irc
from twisted.words.protocols.irc import IRCClient, attributes as A
def _sendISUPPORT(self):
    args = 'MODES=4 CHANLIMIT=#:20 NICKLEN=16 USERLEN=10 HOSTLEN=63 TOPICLEN=450 KICKLEN=450 CHANNELLEN=30 KEYLEN=23 CHANTYPES=# PREFIX=(ov)@+ CASEMAPPING=ascii CAPAB IRCD=dancer'
    msg = 'are available on this server'
    self._serverTestImpl('005', msg, 'isupport', args=args, options=['MODES=4', 'CHANLIMIT=#:20', 'NICKLEN=16', 'USERLEN=10', 'HOSTLEN=63', 'TOPICLEN=450', 'KICKLEN=450', 'CHANNELLEN=30', 'KEYLEN=23', 'CHANTYPES=#', 'PREFIX=(ov)@+', 'CASEMAPPING=ascii', 'CAPAB', 'IRCD=dancer'])