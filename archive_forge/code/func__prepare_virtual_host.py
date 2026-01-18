import asyncio
import json
import logging
import numbers
import socket
import sys
from urllib.parse import quote, unquote, urljoin, urlparse
from tornado import httpclient, ioloop
def _prepare_virtual_host(self, vhost):
    if not isinstance(vhost, numbers.Integral):
        if not vhost or vhost == '/':
            vhost = 0
        elif vhost.startswith('/'):
            vhost = vhost[1:]
        try:
            vhost = int(vhost)
        except ValueError as exc:
            raise ValueError('Database is int between 0 and limit - 1, not {vhost}') from exc
    return vhost