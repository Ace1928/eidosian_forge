import atexit
from ctypes import sizeof
import multiprocessing
import threading
import socket
import time
from cupyx.distributed import _klv_utils
from cupyx.distributed import _store_actions
def _send_recv(self, action):
    for i in range(TCPStoreProxy.MAX_NUM_RETRIES):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((self.host, self.port))
                s.sendall(action.klv())
                result_bytes = s.recv(sizeof(_klv_utils.result_action_t))
                if len(result_bytes) > 0:
                    result = _klv_utils.result_action_t.from_buffer_copy(result_bytes)
                    value = bytearray(result.value)[:result.length]
                    if result.status == 0:
                        return action.decode_result(value)
                    else:
                        raise RuntimeError(value.decode('utf-8'))
        except ConnectionRefusedError:
            time.sleep(TCPStoreProxy.DELAY_FOR_RETRY)
    raise RuntimeError('TCPStore is not available')