import atexit
import os
import re
import signal
import socket
import sys
import warnings
from getpass import getpass, getuser
from multiprocessing import Process
def _split_server(server):
    if '@' in server:
        username, server = server.split('@', 1)
    else:
        username = getuser()
    if ':' in server:
        server, port = server.split(':')
        port = int(port)
    else:
        port = 22
    return (username, server, port)