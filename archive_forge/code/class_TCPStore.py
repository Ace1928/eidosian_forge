import atexit
from ctypes import sizeof
import multiprocessing
import threading
import socket
import time
from cupyx.distributed import _klv_utils
from cupyx.distributed import _store_actions
class TCPStore:

    def __init__(self, world_size):
        self.storage = {}
        self._process = None
        self._world_size = world_size
        self._run = multiprocessing.Value('b', 1)
        self._lock = threading.Lock()
        self._current_barrier = None

    def __del__(self):
        if not _exit_mode:
            self.stop()

    def _set_process(self, process):
        self._process = process

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

    def _server_loop(self, host, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            s.listen()
            s.settimeout(0.5)
            while self._run.value == 1:
                try:
                    c_socket, addr = s.accept()
                except socket.timeout:
                    continue
                t = threading.Thread(target=self._process_request, args=(c_socket,), daemon=True)
                t.start()

    def run(self, host=_DEFAULT_HOST, port=_DEFAULT_PORT):
        p = ExceptionAwareProcess(target=self._server_loop, args=(host, port))
        p.start()
        self._process = p

    def stop(self):
        if _exit_mode:
            return
        if self._process is not None:
            with self._run.get_lock():
                self._run.value = 0
            self._process.join()