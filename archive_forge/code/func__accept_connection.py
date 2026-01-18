import threading
import socket
import select
def _accept_connection(self):
    try:
        ready, _, _ = select.select([self.server_sock], [], [], self.WAIT_EVENT_TIMEOUT)
        if not ready:
            return None
        return self.server_sock.accept()[0]
    except (select.error, socket.error):
        return None