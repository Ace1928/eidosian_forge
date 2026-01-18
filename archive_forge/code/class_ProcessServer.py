import os
import sys
import time
import errno
import socket
import signal
import logging
import threading
import traceback
import email.message
import pyzor.config
import pyzor.account
import pyzor.engines.common
import pyzor.hacks.py26
class ProcessServer(SocketServer.ForkingMixIn, Server):
    """A multi-processing version of the pyzord server.  Each connection is
    served in a new process. This may not be suitable for all database types.
    """

    def __init__(self, address, database, passwd_fn, access_fn, max_children=40, forwarding_server=None):
        ProcessServer.max_children = max_children
        Server.__init__(self, address, database, passwd_fn, access_fn, forwarder=forwarding_server)