import json
import logging as log
from urllib import parse as urlparse
import netaddr
from oslo_concurrency.lockutils import synchronized
import requests
from osprofiler.drivers import base
from osprofiler import exc
def _build_base_url(self, scheme):
    proto_str = '%s://' % scheme
    host_str = '[%s]' % self._host if netaddr.valid_ipv6(self._host) else self._host
    port_str = ':%d' % (self._api_ssl_port if scheme == 'https' else self._api_port)
    return proto_str + host_str + port_str