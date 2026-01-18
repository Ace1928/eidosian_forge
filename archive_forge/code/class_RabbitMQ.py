import asyncio
import json
import logging
import numbers
import socket
import sys
from urllib.parse import quote, unquote, urljoin, urlparse
from tornado import httpclient, ioloop
class RabbitMQ(BrokerBase):

    def __init__(self, broker_url, http_api, io_loop=None, **__):
        super().__init__(broker_url)
        self.io_loop = io_loop or ioloop.IOLoop.instance()
        self.host = self.host or 'localhost'
        self.port = self.port or 15672
        self.vhost = quote(self.vhost, '') or '/' if self.vhost != '/' else self.vhost
        self.username = self.username or 'guest'
        self.password = self.password or 'guest'
        if not http_api:
            http_api = f'http://{self.username}:{self.password}@{self.host}:{self.port}/api/{self.vhost}'
        try:
            self.validate_http_api(http_api)
        except ValueError:
            logger.error('Invalid broker api url: %s', http_api)
        self.http_api = http_api

    async def queues(self, names):
        url = urljoin(self.http_api, 'queues/' + self.vhost)
        api_url = urlparse(self.http_api)
        username = unquote(api_url.username or '') or self.username
        password = unquote(api_url.password or '') or self.password
        http_client = httpclient.AsyncHTTPClient()
        try:
            response = await http_client.fetch(url, auth_username=username, auth_password=password, connect_timeout=1.0, request_timeout=2.0, validate_cert=False)
        except (socket.error, httpclient.HTTPError) as e:
            logger.error('RabbitMQ management API call failed: %s', e)
            return []
        finally:
            http_client.close()
        if response.code == 200:
            info = json.loads(response.body.decode())
            return [x for x in info if x['name'] in names]
        response.rethrow()

    @classmethod
    def validate_http_api(cls, http_api):
        url = urlparse(http_api)
        if url.scheme not in ('http', 'https'):
            raise ValueError(f'Invalid http api schema: {url.scheme}')