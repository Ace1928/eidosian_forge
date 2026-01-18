import asyncio
import json
import logging
import numbers
import socket
import sys
from urllib.parse import quote, unquote, urljoin, urlparse
from tornado import httpclient, ioloop
def _get_redis_client_args(self):
    client_args = super()._get_redis_client_args()
    client_args['ssl'] = True
    if isinstance(self.broker_use_ssl, dict):
        client_args.update(self.broker_use_ssl)
    return client_args