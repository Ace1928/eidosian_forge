import http.client
import os
import sys
import time
import httplib2
from oslo_config import cfg
from oslo_serialization import jsonutils
from oslo_utils.fixture import uuidsentinel as uuids
from glance import context
import glance.db as db_api
from glance.tests import functional
from glance.tests.utils import execute
def _send_http_request(self, path, method):
    headers = {'Content-Type': 'application/json'}
    return httplib2.Http().request(path, method, None, self._headers(headers))