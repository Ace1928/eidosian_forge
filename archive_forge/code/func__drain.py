import os
import socket
import asyncio
def _drain(self):
    while not self._queue.empty():
        msg = self._queue.get_nowait()
        self._queue.task_done()
        try:
            self._send(msg)
        except BlockingIOError:
            self._monitor = True
            self._loop.add_writer(self._sock.fileno(), self._drain)
            break
        except OSError:
            pass
    else:
        if self._monitor:
            self._monitor = False
            self._loop.remove_writer(self._sock.fileno())