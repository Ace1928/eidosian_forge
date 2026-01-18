import math
import re
from io import BytesIO
from struct import calcsize, pack, unpack
from zope.interface import implementer
from twisted.internet import defer, interfaces, protocol
from twisted.python import log
class StatefulStringProtocol:
    """
    A stateful string protocol.

    This is a mixin for string protocols (L{Int32StringReceiver},
    L{NetstringReceiver}) which translates L{stringReceived} into a callback
    (prefixed with C{'proto_'}) depending on state.

    The state C{'done'} is special; if a C{proto_*} method returns it, the
    connection will be closed immediately.

    @ivar state: Current state of the protocol. Defaults to C{'init'}.
    @type state: C{str}
    """
    state = 'init'

    def stringReceived(self, string):
        """
        Choose a protocol phase function and call it.

        Call back to the appropriate protocol phase; this begins with
        the function C{proto_init} and moves on to C{proto_*} depending on
        what each C{proto_*} function returns.  (For example, if
        C{self.proto_init} returns 'foo', then C{self.proto_foo} will be the
        next function called when a protocol message is received.
        """
        try:
            pto = 'proto_' + self.state
            statehandler = getattr(self, pto)
        except AttributeError:
            log.msg('callback', self.state, 'not found')
        else:
            self.state = statehandler(string)
            if self.state == 'done':
                self.transport.loseConnection()