import asyncio
import json
import logging
import numbers
import socket
import sys
from urllib.parse import quote, unquote, urljoin, urlparse
from tornado import httpclient, ioloop
class RedisSentinel(RedisBase):

    def __init__(self, broker_url, *args, **kwargs):
        super().__init__(broker_url, *args, **kwargs)
        broker_options = kwargs.get('broker_options', {})
        self.host = self.host or 'localhost'
        self.port = self.port or 26379
        self.vhost = self._prepare_virtual_host(self.vhost)
        self.master_name = self._prepare_master_name(broker_options)
        self.redis = self._get_redis_client(broker_options)

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

    def _prepare_master_name(self, broker_options):
        try:
            master_name = broker_options['master_name']
        except KeyError as exc:
            raise ValueError('master_name is required for Sentinel broker') from exc
        return master_name

    def _get_redis_client(self, broker_options):
        connection_kwargs = {'password': self.password, 'sentinel_kwargs': broker_options.get('sentinel_kwargs')}
        sentinel = redis.sentinel.Sentinel([(self.host, self.port)], **connection_kwargs)
        redis_client = sentinel.master_for(self.master_name)
        return redis_client