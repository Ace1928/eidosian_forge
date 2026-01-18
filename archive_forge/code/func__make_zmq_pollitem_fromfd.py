import warnings
from zmq.error import InterruptedSystemCall, _check_rc
from ._cffi import ffi
from ._cffi import lib as C
def _make_zmq_pollitem_fromfd(socket_fd, flags):
    zmq_pollitem = ffi.new('zmq_pollitem_t*')
    zmq_pollitem.socket = ffi.NULL
    zmq_pollitem.fd = socket_fd
    zmq_pollitem.events = flags
    zmq_pollitem.revents = 0
    return zmq_pollitem[0]