import json
import logging as log
from urllib import parse as urlparse
import netaddr
from oslo_concurrency.lockutils import synchronized
import requests
from osprofiler.drivers import base
from osprofiler import exc
def _trunc_session_id(self):
    if self._session_id:
        return self._session_id[-5:]