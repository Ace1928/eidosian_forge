import json
import logging as log
from urllib import parse as urlparse
import netaddr
from oslo_concurrency.lockutils import synchronized
import requests
from osprofiler.drivers import base
from osprofiler import exc
def _create_field(name, content):
    return {'name': name, 'content': content}