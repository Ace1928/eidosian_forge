import http.client as http_client
import eventlet.patcher
import httplib2
import webob.dec
import webob.exc
from glance.common import client
from glance.common import exception
from glance.common import wsgi
from glance.tests import functional
from glance.tests import utils
def _do_test_exception(self, path, exc_type):
    try:
        self.client.do_request('GET', path)
        self.fail('expected %s' % exc_type)
    except exc_type as e:
        if 'retry' in path:
            self.assertEqual(10, e.retry_after)