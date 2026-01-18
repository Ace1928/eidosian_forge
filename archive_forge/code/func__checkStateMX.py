import email.utils
import os
import pickle
import time
from typing import Type
from twisted.application import internet
from twisted.internet import protocol
from twisted.internet.defer import Deferred, DeferredList
from twisted.internet.error import DNSLookupError
from twisted.internet.protocol import connectionDone
from twisted.mail import bounce, relay, smtp
from twisted.python import log
from twisted.python.failure import Failure
def _checkStateMX(self):
    nextMessages = self.queue.getWaiting()
    nextMessages.reverse()
    exchanges = {}
    for msg in nextMessages:
        from_, to = self.queue.getEnvelope(msg)
        name, addr = email.utils.parseaddr(to)
        parts = addr.split('@', 1)
        if len(parts) != 2:
            log.err('Illegal message destination: ' + to)
            continue
        domain = parts[1]
        self.queue.setRelaying(msg)
        exchanges.setdefault(domain, []).append(self.queue.getPath(msg))
        if len(exchanges) >= self.maxConnections - len(self.managed):
            break
    if self.mxcalc is None:
        self.mxcalc = MXCalculator()
    relays = []
    for domain, msgs in exchanges.iteritems():
        manager = _AttemptManager(self, self.queue.noisy)
        factory = self.factory(msgs, manager, *self.fArgs, **self.fKwArgs)
        self.managed[factory] = map(os.path.basename, msgs)
        relayAttemptDeferred = manager.getCompletionDeferred()
        connectSetupDeferred = self.mxcalc.getMX(domain)
        connectSetupDeferred.addCallback(lambda mx: str(mx.name))
        connectSetupDeferred.addCallback(self._cbExchange, self.PORT, factory)
        connectSetupDeferred.addErrback(lambda err: (relayAttemptDeferred.errback(err), err)[1])
        connectSetupDeferred.addErrback(self._ebExchange, factory, domain)
        relays.append(relayAttemptDeferred)
    return DeferredList(relays)