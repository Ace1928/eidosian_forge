import asyncio
import json
import logging
import numbers
import socket
import sys
from urllib.parse import quote, unquote, urljoin, urlparse
from tornado import httpclient, ioloop
class RedisSocket(RedisBase):

    def __init__(self, broker_url, *args, **kwargs):
        super().__init__(broker_url, *args, **kwargs)
        self.redis = redis.Redis(unix_socket_path='/' + self.vhost, password=self.password)