import errno
import socket
from pathlib import Path
from threading import Thread
import zmq
from jupyter_client.localinterfaces import localhost
def _try_bind_socket(self):
    c = ':' if self.transport == 'tcp' else '-'
    return self.socket.bind(f'{self.transport}://{self.ip}' + c + str(self.port))