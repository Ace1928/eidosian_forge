from __future__ import print_function
import logging
import os
import random
import socket
import sys
import time
def _is_port_free(port, return_sockets=None):
    """Internal implementation of is_port_free.

    Args:
      port: integer, port to check
      return_sockets: If supplied, a list that we will append open bound
        sockets on the port in question to rather than closing them.

    Returns:
      bool, whether port is free to use for both TCP and UDP.
    """
    return _bind(port, *_PROTOS[0], return_socket=return_sockets) and _bind(port, *_PROTOS[1], return_socket=return_sockets)