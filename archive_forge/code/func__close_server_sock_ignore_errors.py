import threading
import socket
import select
def _close_server_sock_ignore_errors(self):
    try:
        self.server_sock.close()
    except IOError:
        pass