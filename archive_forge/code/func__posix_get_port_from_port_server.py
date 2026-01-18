from __future__ import print_function
import logging
import os
import random
import socket
import sys
import time
def _posix_get_port_from_port_server(portserver_address, pid):
    if portserver_address[0] == '@':
        portserver_address = '\x00' + portserver_address[1:]
    try:
        if hasattr(socket, 'AF_UNIX'):
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        else:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect(portserver_address)
            sock.sendall(('%d\n' % pid).encode('ascii'))
            return sock.recv(1024)
        finally:
            sock.close()
    except socket.error as error:
        print('Socket error when connecting to portserver:', error, file=sys.stderr)
        return None