import email.parser
import email.message
import errno
import http
import io
import re
import socket
import sys
import collections.abc
from urllib.parse import urlsplit
def _get_hostport(self, host, port):
    if port is None:
        i = host.rfind(':')
        j = host.rfind(']')
        if i > j:
            try:
                port = int(host[i + 1:])
            except ValueError:
                if host[i + 1:] == '':
                    port = self.default_port
                else:
                    raise InvalidURL("nonnumeric port: '%s'" % host[i + 1:])
            host = host[:i]
        else:
            port = self.default_port
        if host and host[0] == '[' and (host[-1] == ']'):
            host = host[1:-1]
    return (host, port)