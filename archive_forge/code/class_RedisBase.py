import asyncio
import json
import logging
import numbers
import socket
import sys
from urllib.parse import quote, unquote, urljoin, urlparse
from tornado import httpclient, ioloop
class RedisBase(BrokerBase):
    DEFAULT_SEP = '\x06\x16'
    DEFAULT_PRIORITY_STEPS = [0, 3, 6, 9]

    def __init__(self, broker_url, *_, **kwargs):
        super().__init__(broker_url)
        self.redis = None
        if not redis:
            raise ImportError('redis library is required')
        broker_options = kwargs.get('broker_options', {})
        self.priority_steps = broker_options.get('priority_steps', self.DEFAULT_PRIORITY_STEPS)
        self.sep = broker_options.get('sep', self.DEFAULT_SEP)
        self.broker_prefix = broker_options.get('global_keyprefix', '')

    def _q_for_pri(self, queue, pri):
        if pri not in self.priority_steps:
            raise ValueError('Priority not in priority steps')
        return '{0}{1}{2}'.format(*((queue, self.sep, pri) if pri else (queue, '', '')))

    async def queues(self, names):
        queue_stats = []
        for name in names:
            priority_names = [self.broker_prefix + self._q_for_pri(name, pri) for pri in self.priority_steps]
            queue_stats.append({'name': name, 'messages': sum((self.redis.llen(x) for x in priority_names))})
        return queue_stats