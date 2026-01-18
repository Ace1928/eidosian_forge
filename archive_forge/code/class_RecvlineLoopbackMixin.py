import os
import sys
from unittest import skipIf
from twisted.conch import recvline
from twisted.conch.insults import insults
from twisted.cred import portal
from twisted.internet import defer, error
from twisted.internet.testing import StringTransport
from twisted.python import components, filepath, reflect
from twisted.python.compat import iterbytes
from twisted.python.reflect import requireModule
from twisted.trial.unittest import SkipTest, TestCase
from twisted.conch import telnet
from twisted.conch.insults import helper
from twisted.conch.test.loopback import LoopbackRelay
from twisted.cred import checkers
from twisted.conch.test import test_telnet
class RecvlineLoopbackMixin:
    serverProtocol = EchoServer

    def testSimple(self):
        return self._trivialTest(b'first line\ndone', [b'>>> first line', b'first line', b'>>> done'])

    def testLeftArrow(self):
        return self._trivialTest(insert + b'first line' + left * 4 + b'xxxx\ndone', [b'>>> first xxxx', b'first xxxx', b'>>> done'])

    def testRightArrow(self):
        return self._trivialTest(insert + b'right line' + left * 4 + right * 2 + b'xx\ndone', [b'>>> right lixx', b'right lixx', b'>>> done'])

    def testBackspace(self):
        return self._trivialTest(b'second line' + backspace * 4 + b'xxxx\ndone', [b'>>> second xxxx', b'second xxxx', b'>>> done'])

    def testDelete(self):
        return self._trivialTest(b'delete xxxx' + left * 4 + delete * 4 + b'line\ndone', [b'>>> delete line', b'delete line', b'>>> done'])

    def testInsert(self):
        return self._trivialTest(b'third ine' + left * 3 + b'l\ndone', [b'>>> third line', b'third line', b'>>> done'])

    def testTypeover(self):
        return self._trivialTest(b'fourth xine' + left * 4 + insert + b'l\ndone', [b'>>> fourth line', b'fourth line', b'>>> done'])

    def testHome(self):
        return self._trivialTest(insert + b'blah line' + home + b'home\ndone', [b'>>> home line', b'home line', b'>>> done'])

    def testEnd(self):
        return self._trivialTest(b'end ' + left * 4 + end + b'line\ndone', [b'>>> end line', b'end line', b'>>> done'])