import asyncio
import json
import logging
import numbers
import socket
import sys
from urllib.parse import quote, unquote, urljoin, urlparse
from tornado import httpclient, ioloop
class BrokerBase:

    def __init__(self, broker_url, *_, **__):
        purl = urlparse(broker_url)
        self.host = purl.hostname
        self.port = purl.port
        self.vhost = purl.path[1:]
        username = purl.username
        password = purl.password
        self.username = unquote(username) if username else username
        self.password = unquote(password) if password else password

    async def queues(self, names):
        raise NotImplementedError