import atexit
from ctypes import sizeof
import multiprocessing
import threading
import socket
import time
from cupyx.distributed import _klv_utils
from cupyx.distributed import _store_actions
def _process_request(self, c_socket):
    with c_socket:
        action_bytes = c_socket.recv(sizeof(_klv_utils.action_t))
        if len(action_bytes) > 0:
            action_m = _klv_utils.action_t.from_buffer_copy(action_bytes)
            if action_m.length > 256:
                raise ValueError('Invalid length for message')
            value = bytearray(action_m.value)[:action_m.length]
            r = _store_actions.execute_action(action_m.action, value, self)
            if r is not None:
                c_socket.sendall(r.klv())