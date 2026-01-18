import re
from hashlib import md5
from typing import List
from twisted.internet import defer, error, interfaces
from twisted.mail._except import (
from twisted.protocols import basic, policies
from twisted.python import log
def _consumeOrSetItem(self, cmd, args, consumer, xform):
    """
        Send a command to which a long response is expected and process the
        multi-line response into a list accounting for deleted messages.

        @type cmd: L{bytes}
        @param cmd: A POP3 command to which a long response is expected.

        @type args: L{bytes}
        @param args: The command arguments.

        @type consumer: L{None} or callable that takes
            L{object}
        @param consumer: L{None} or a function that consumes the output from
            the transform function.

        @type xform: L{None}, callable that takes
            L{bytes} and returns 2-L{tuple} of (0) L{int}, (1) L{object},
            or callable that takes L{bytes} and returns L{object}
        @param xform: A function that parses a line from a multi-line response
            and transforms the values into usable form for input to the
            consumer function.  If no consumer function is specified, the
            output must be a message index and corresponding value.  If no
            transform function is specified, the line is used as is.

        @rtype: L{Deferred <defer.Deferred>} which fires with L{list} of
            L{object} or callable that takes L{list} of L{object}
        @return: A deferred which fires when the entire response has been
            received.  When a consumer is not provided, the return value is a
            list of the value for each message or L{None} for deleted messages.
            Otherwise, it returns the consumer itself.
        """
    if consumer is None:
        L = []
        consumer = _ListSetter(L).setitem
        return self.sendLong(cmd, args, consumer, xform).addCallback(lambda r: L)
    return self.sendLong(cmd, args, consumer, xform)