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
def _send_create_image_http_request(self, path, body=None):
    headers = {'Content-Type': 'application/json', 'X-Roles': 'admin'}
    body = body or {'container_format': 'ovf', 'disk_format': 'raw', 'name': 'test_image', 'visibility': 'public'}
    body = jsonutils.dumps(body)
    return httplib2.Http().request(path, 'POST', body, self._headers(headers))