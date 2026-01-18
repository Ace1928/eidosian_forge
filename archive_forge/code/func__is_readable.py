import select
import socket
from .parser import Parser
from .ports import BaseIOPort, MultiPort
def _is_readable(socket):
    """Return True if there is data to be read on the socket."""
    timeout = 0
    rlist, wlist, elist = select.select([socket.fileno()], [], [], timeout)
    return bool(rlist)