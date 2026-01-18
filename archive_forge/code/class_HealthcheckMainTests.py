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
class HealthcheckMainTests(test_base.BaseTestCase):

    def test_startup_response(self):
        server = __main__.create_server(0)
        th = threading.Thread(target=server.serve_forever)
        th.start()
        self.addCleanup(server.shutdown)
        while True:
            try:
                r = requests.get('http://127.0.0.1:%s' % server.server_address[1], timeout=10)
            except requests.ConnectionError:
                time.sleep(1)
            else:
                self.assertEqual(200, r.status_code)
                break