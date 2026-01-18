from typing import Callable
import zmq
from zmq.backend import monitored_queue as _backend_mq
def _monitored_queue(in_socket, out_socket, mon_socket, in_prefix=b'in', out_prefix=b'out'):
    swap_ids = in_socket.type == zmq.ROUTER and out_socket.type == zmq.ROUTER
    poller = zmq.Poller()
    poller.register(in_socket, zmq.POLLIN)
    poller.register(out_socket, zmq.POLLIN)
    while True:
        events = dict(poller.poll())
        if in_socket in events:
            _relay(in_socket, out_socket, mon_socket, in_prefix, swap_ids)
        if out_socket in events:
            _relay(out_socket, in_socket, mon_socket, out_prefix, swap_ids)