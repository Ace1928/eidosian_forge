import select
import socket
from .parser import Parser
from .ports import BaseIOPort, MultiPort
class SocketPort(BaseIOPort):

    def __init__(self, host, portno, conn=None):
        BaseIOPort.__init__(self, name=format_address(host, portno))
        self.closed = False
        self._parser = Parser()
        self._messages = self._parser.messages
        if conn is None:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.setblocking(True)
            self._socket.connect((host, portno))
        else:
            self._socket = conn
        kwargs = {'buffering': 0}
        self._rfile = self._socket.makefile('rb', **kwargs)
        self._wfile = self._socket.makefile('wb', **kwargs)

    def _get_device_type(self):
        return 'socket'

    def _receive(self, block=True):
        while _is_readable(self._socket):
            try:
                byte = self._rfile.read(1)
            except OSError as err:
                raise OSError(err.args[1]) from err
            if len(byte) == 0:
                self.close()
                break
            else:
                self._parser.feed_byte(ord(byte))

    def _send(self, message):
        try:
            self._wfile.write(message.bin())
            self._wfile.flush()
        except OSError as err:
            if err.errno == 32:
                self.close()
            raise OSError(err.args[1]) from err

    def _close(self):
        self._socket.close()