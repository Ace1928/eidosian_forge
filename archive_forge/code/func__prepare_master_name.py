import asyncio
import json
import logging
import numbers
import socket
import sys
from urllib.parse import quote, unquote, urljoin, urlparse
from tornado import httpclient, ioloop
def _prepare_master_name(self, broker_options):
    try:
        master_name = broker_options['master_name']
    except KeyError as exc:
        raise ValueError('master_name is required for Sentinel broker') from exc
    return master_name