import contextlib
import logging
import os
import socket
import sys
def _abstractify(socket_name):
    if socket_name.startswith('@'):
        socket_name = '\x00%s' % socket_name[1:]
    return socket_name