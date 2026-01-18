import socket
from incremental import Version
from twisted.python import deprecate
class CannotListenError(BindError):
    """
    This gets raised by a call to startListening, when the object cannotstart
    listening.

    @ivar interface: the interface I tried to listen on
    @ivar port: the port I tried to listen on
    @ivar socketError: the exception I got when I tried to listen
    @type socketError: L{socket.error}
    """

    def __init__(self, interface, port, socketError):
        BindError.__init__(self, interface, port, socketError)
        self.interface = interface
        self.port = port
        self.socketError = socketError

    def __str__(self) -> str:
        iface = self.interface or 'any'
        return "Couldn't listen on {}:{}: {}.".format(iface, self.port, self.socketError)