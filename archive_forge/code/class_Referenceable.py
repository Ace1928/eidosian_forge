import sys
from zope.interface import Interface, implementer
from twisted.python import log, reflect
from twisted.python.compat import cmp, comparable
from .jelly import (
class Referenceable(Serializable):
    perspective = None
    'I am an object sent remotely as a direct reference.\n\n    When one of my subclasses is sent as an argument to or returned\n    from a remote method call, I will be serialized by default as a\n    direct reference.\n\n    This means that the peer will be able to call methods on me;\n    a method call xxx() from my peer will be resolved to methods\n    of the name remote_xxx.\n    '

    def remoteMessageReceived(self, broker, message, args, kw):
        """A remote message has been received.  Dispatch it appropriately.

        The default implementation is to dispatch to a method called
        'remote_messagename' and call it with the same arguments.
        """
        args = broker.unserialize(args)
        kw = broker.unserialize(kw)
        if [key for key in kw.keys() if isinstance(key, bytes)]:
            kw = {k.decode('utf8'): v for k, v in kw.items()}
        if not isinstance(message, str):
            message = message.decode('utf8')
        method = getattr(self, 'remote_%s' % message, None)
        if method is None:
            raise NoSuchMethod(f'No such method: remote_{message}')
        try:
            state = method(*args, **kw)
        except TypeError:
            log.msg(f"{method} didn't accept {args} and {kw}")
            raise
        return broker.serialize(state, self.perspective)

    def jellyFor(self, jellier):
        """(internal)

        Return a tuple which will be used as the s-expression to
        serialize this to a peer.
        """
        return [b'remote', jellier.invoker.registerReference(self)]