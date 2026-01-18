from __future__ import print_function
import logging
import os
import random
import socket
import sys
import time
def _windows_get_port_from_port_server(portserver_address, pid):
    if portserver_address[0] == '@':
        portserver_address = '\\\\.\\pipe\\' + portserver_address[1:]
    try:
        handle = _winapi.CreateFile(portserver_address, _winapi.GENERIC_READ | _winapi.GENERIC_WRITE, 0, 0, _winapi.OPEN_EXISTING, 0, 0)
        _winapi.WriteFile(handle, ('%d\n' % pid).encode('ascii'))
        data, _ = _winapi.ReadFile(handle, 6, 0)
        return data
    except FileNotFoundError as error:
        print('File error when connecting to portserver:', error, file=sys.stderr)
        return None