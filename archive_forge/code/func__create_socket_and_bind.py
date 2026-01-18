import threading
import socket
import select
def _create_socket_and_bind(self):
    sock = socket.socket()
    sock.bind((self.host, self.port))
    sock.listen(5)
    return sock