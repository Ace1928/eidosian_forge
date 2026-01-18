from contextlib import closing, contextmanager
import errno
import socket
import threading
import time
import http.client
import pytest
import cheroot.server
from cheroot.test import webtest
import cheroot.wsgi
def _get_conn_data(bind_addr):
    if isinstance(bind_addr, tuple):
        host, port = bind_addr
    else:
        host, port = (bind_addr, 0)
    interface = webtest.interface(host)
    if ':' in interface and (not _probe_ipv6_sock(interface)):
        interface = '127.0.0.1'
        if ':' in host:
            host = interface
    return (interface, host, port)