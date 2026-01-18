import re
from zope.interface import implementer
from twisted.internet.defer import (
from twisted.internet.error import ConnectionDone
from twisted.internet.interfaces import IConsumer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.logger import Logger
from twisted.protocols.basic import LineReceiver
from twisted.python.compat import networkString
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.python.reflect import fullyQualifiedName
from twisted.web.http import (
from twisted.web.http_headers import Headers
from twisted.web.iweb import UNKNOWN_LENGTH, IClientRequest, IResponse
def _writeToBodyProducerContentLength(self, transport):
    """
        Write this request to the given transport using content-length to frame
        the body.

        @param transport: See L{writeTo}.
        @return: See L{writeTo}.
        """
    self._writeHeaders(transport, networkString('Content-Length: %d\r\n' % (self.bodyProducer.length,)))
    finishedConsuming = Deferred()
    encoder = LengthEnforcingConsumer(self.bodyProducer, transport, finishedConsuming)
    transport.registerProducer(self.bodyProducer, True)
    finishedProducing = self.bodyProducer.startProducing(encoder)

    def combine(consuming, producing):

        def cancelConsuming(ign):
            finishedProducing.cancel()
        ultimate = Deferred(cancelConsuming)
        state = [None]

        def ebConsuming(err):
            if state == [None]:
                state[0] = 1
                ultimate.errback(err)
            else:
                self._log.failure('Buggy state machine in {request}/[{state}]: ebConsuming called', failure=err, request=repr(self), state=state[0])

        def cbProducing(result):
            if state == [None]:
                state[0] = 2
                try:
                    encoder._noMoreWritesExpected()
                except BaseException:
                    ultimate.errback()
                else:
                    ultimate.callback(None)

        def ebProducing(err):
            if state == [None]:
                state[0] = 3
                encoder._allowNoMoreWrites()
                ultimate.errback(err)
            else:
                self._log.failure('Producer is buggy', failure=err)
        consuming.addErrback(ebConsuming)
        producing.addCallbacks(cbProducing, ebProducing)
        return ultimate
    d = combine(finishedConsuming, finishedProducing)

    def f(passthrough):
        transport.unregisterProducer()
        return passthrough
    d.addBoth(f)
    return d