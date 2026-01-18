import sys
import warnings
from functools import wraps
from io import BytesIO
from twisted.internet import defer, protocol
from twisted.python import failure
def _callProtocolWithDeferred(protocol, executable, args, env, path, reactor=None, protoArgs=()):
    if reactor is None:
        from twisted.internet import reactor
    d = defer.Deferred()
    p = protocol(d, *protoArgs)
    reactor.spawnProcess(p, executable, (executable,) + tuple(args), env, path)
    return d