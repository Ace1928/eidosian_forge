import errno
from threading import Event
import zmq
import zmq.error
from zmq.constants import ETERM
from ._cffi import ffi
from ._cffi import lib as C
def _buffer_from_zmq_msg(self):
    """one-time extract buffer from zmq_msg

        for Frames created by recv
        """
    if self._data is None:
        self._data = ffi.buffer(C.zmq_msg_data(self.zmq_msg), C.zmq_msg_size(self.zmq_msg))
    if self._buffer is None:
        self._buffer = memoryview(self._data)