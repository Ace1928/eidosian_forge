import datetime
import gettext
import http.client as http
import os
import socket
from unittest import mock
import eventlet.patcher
import fixtures
from oslo_concurrency import processutils
from oslo_serialization import jsonutils
import routes
import webob
from glance.api.v2 import router as router_v2
from glance.common import exception
from glance.common import utils
from glance.common import wsgi
from glance import i18n
from glance.image_cache import prefetcher
from glance.tests import utils as test_utils
def _check_transfer_encoding(self, transfer_encoding=None, content_length=None, include_body=True):
    request = wsgi.Request.blank('/')
    request.method = 'POST'
    if include_body:
        request.body = b'fake_body'
    request.headers['transfer-encoding'] = transfer_encoding
    if content_length is not None:
        request.headers['content-length'] = content_length
    return wsgi.JSONRequestDeserializer().has_body(request)