import re
import string
from zope.interface import implementer
from incremental import Version
from twisted.conch.insults import insults
from twisted.internet import defer, protocol, reactor
from twisted.logger import Logger
from twisted.python import _textattributes
from twisted.python.compat import iterbytes
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
class ExpectableBuffer(TerminalBuffer):
    _mark = 0

    def connectionMade(self):
        TerminalBuffer.connectionMade(self)
        self._expecting = []

    def write(self, data):
        TerminalBuffer.write(self, data)
        self._checkExpected()

    def cursorHome(self):
        TerminalBuffer.cursorHome(self)
        self._mark = 0

    def _timeoutExpected(self, d):
        d.errback(ExpectationTimeout())
        self._checkExpected()

    def _checkExpected(self):
        s = self.__bytes__()[self._mark:]
        while self._expecting:
            expr, timer, deferred = self._expecting[0]
            if timer and (not timer.active()):
                del self._expecting[0]
                continue
            for match in expr.finditer(s):
                if timer:
                    timer.cancel()
                del self._expecting[0]
                self._mark += match.end()
                s = s[match.end():]
                deferred.callback(match)
                break
            else:
                return

    def expect(self, expression, timeout=None, scheduler=reactor):
        d = defer.Deferred()
        timer = None
        if timeout:
            timer = scheduler.callLater(timeout, self._timeoutExpected, d)
        self._expecting.append((re.compile(expression), timer, d))
        self._checkExpected()
        return d