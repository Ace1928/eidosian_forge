import select
import socket
from .parser import Parser
from .ports import BaseIOPort, MultiPort
class PortServer(MultiPort):

    def __init__(self, host, portno, backlog=1):
        MultiPort.__init__(self, format_address(host, portno))
        self.ports = []
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)
        self._socket.setblocking(True)
        self._socket.bind((host, portno))
        self._socket.listen(backlog)

    def _get_device_type(self):
        return 'server'

    def _close(self):
        for port in self.ports:
            port.close()
        self._socket.close()

    def _update_ports(self):
        """Remove closed port ports."""
        self.ports = [port for port in self.ports if not port.closed]

    def accept(self, block=True):
        """
        Accept a connection from a client.

        Will block until there is a new connection, and then return a
        SocketPort object.

        If block=False, None will be returned if there is no
        new connection waiting.
        """
        if not block and (not _is_readable(self._socket)):
            return None
        self._update_ports()
        conn, (host, port) = self._socket.accept()
        return SocketPort(host, port, conn=conn)

    def _send(self, message):
        self._update_ports()
        return MultiPort._send(self, message)

    def _receive(self, block=True):
        port = self.accept(block=False)
        if port:
            self.ports.append(port)
        self._update_ports()
        return MultiPort._receive(self)