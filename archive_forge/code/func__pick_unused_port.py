from __future__ import print_function
import logging
import os
import random
import socket
import sys
import time
def _pick_unused_port(pid=None, portserver_address=None, noserver_bind_timeout=0):
    """Internal implementation of pick_unused_port.

    Args:
      pid, portserver_address: See pick_unused_port().
      noserver_bind_timeout: If no portserver was used, this is the number of
        seconds we will attempt to keep a child process around with the ports
        returned open and bound SO_REUSEADDR style to help avoid race condition
        port reuse. A non-zero value attempts os.fork(). Do not use it in a
        multithreaded process.
    """
    try:
        port = _free_ports.pop()
    except KeyError:
        pass
    else:
        _owned_ports.add(port)
        return port
    if portserver_address:
        port = get_port_from_port_server(portserver_address, pid=pid)
        if port:
            return port
    if 'PORTSERVER_ADDRESS' in os.environ:
        port = get_port_from_port_server(os.environ['PORTSERVER_ADDRESS'], pid=pid)
        if port:
            return port
    return _pick_unused_port_without_server(bind_timeout=noserver_bind_timeout)