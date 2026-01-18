from __future__ import print_function
import logging
import os
import random
import socket
import sys
import time
def _pick_unused_port_without_server(bind_timeout=0):
    """Pick an available network port without the help of a port server.

    This code ensures that the port is available on both TCP and UDP.

    This function is an implementation detail of PickUnusedPort(), and
    should not be called by code outside of this module.

    Args:
      bind_timeout: number of seconds to attempt to keep a child process
          process around bound SO_REUSEADDR style to the port. If we cannot
          do that we emit a warning to stderr.

    Returns:
      A port number that is unused on both TCP and UDP.

    Raises:
      NoFreePortFoundError: No free port could be found.
    """
    port = None
    bound_sockets = [] if bind_timeout > 0 else None
    for _ in range(10):
        port = _bind(0, socket.SOCK_STREAM, socket.IPPROTO_TCP, bound_sockets)
        if port and port not in _random_ports and _bind(port, socket.SOCK_DGRAM, socket.IPPROTO_UDP, bound_sockets):
            _random_ports.add(port)
            _spawn_bound_port_holding_daemon(port, bound_sockets, bind_timeout)
            return port
        if bound_sockets:
            for held_socket in bound_sockets:
                held_socket.close()
            del bound_sockets[:]
    rng = random.Random()
    for _ in range(10):
        port = int(rng.randrange(15000, 25000))
        if port not in _random_ports:
            if _is_port_free(port, bound_sockets):
                _random_ports.add(port)
                _spawn_bound_port_holding_daemon(port, bound_sockets, bind_timeout)
                return port
            if bound_sockets:
                for held_socket in bound_sockets:
                    held_socket.close()
                del bound_sockets[:]
    raise NoFreePortFoundError()