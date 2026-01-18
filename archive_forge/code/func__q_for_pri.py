import asyncio
import json
import logging
import numbers
import socket
import sys
from urllib.parse import quote, unquote, urljoin, urlparse
from tornado import httpclient, ioloop
def _q_for_pri(self, queue, pri):
    if pri not in self.priority_steps:
        raise ValueError('Priority not in priority steps')
    return '{0}{1}{2}'.format(*((queue, self.sep, pri) if pri else (queue, '', '')))