from __future__ import print_function
import logging
import os
import random
import socket
import sys
import time
def _spawn_bound_port_holding_daemon(port, bound_sockets, timeout):
    """If possible, fork()s a daemon process to hold bound_sockets open.

    Emits a warning to stderr if it cannot.

    Args:
      port: The port number the sockets are bound to (informational).
      bound_sockets: The list of bound sockets our child process will hold
          open. If the list is empty, no action is taken.
      timeout: A positive number of seconds the child should sleep for before
          closing the sockets and exiting.
    """
    if bound_sockets and timeout > 0:
        try:
            fork_pid = os.fork()
        except Exception as err:
            print('WARNING: Cannot timeout unbinding close of port', port, ' closing on exit. -', err, file=sys.stderr)
        else:
            if fork_pid == 0:
                try:
                    os.close(sys.stdin.fileno())
                    os.close(sys.stdout.fileno())
                    os.close(sys.stderr.fileno())
                    time.sleep(timeout)
                    for held_socket in bound_sockets:
                        held_socket.close()
                finally:
                    os._exit(0)