from sys import intern
from typing import Type
from twisted.internet import protocol
from twisted.python import failure
from twisted.words.xish import domish, utility
class XmlStreamFactoryMixin(BootstrapMixin):
    """
    XmlStream factory mixin that takes care of event handlers.

    All positional and keyword arguments passed to create this factory are
    passed on as-is to the protocol.

    @ivar args: Positional arguments passed to the protocol upon instantiation.
    @type args: C{tuple}.
    @ivar kwargs: Keyword arguments passed to the protocol upon instantiation.
    @type kwargs: C{dict}.
    """

    def __init__(self, *args, **kwargs):
        BootstrapMixin.__init__(self)
        self.args = args
        self.kwargs = kwargs

    def buildProtocol(self, addr):
        """
        Create an instance of XmlStream.

        The returned instance will have bootstrap event observers registered
        and will proceed to handle input on an incoming connection.
        """
        xs = self.protocol(*self.args, **self.kwargs)
        xs.factory = self
        self.installBootstraps(xs)
        return xs