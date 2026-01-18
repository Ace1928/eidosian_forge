import threading
import time
from unittest import mock
from oslo_config import fixture as config
from oslo_serialization import jsonutils
from oslotest import base as test_base
import requests
import webob.dec
import webob.exc
from oslo_middleware import healthcheck
from oslo_middleware.healthcheck import __main__
def _do_test_request(self, conf={}, path='/healthcheck', accept='text/plain', method='GET', server_port=80, headers=None, remote_addr='127.0.0.1'):
    self.app = healthcheck.Healthcheck(self.application, conf)
    req = webob.Request.blank(path, accept=accept, method=method)
    req.server_port = server_port
    if headers:
        req.headers = headers
    req.remote_addr = remote_addr
    res = req.get_response(self.app)
    return res