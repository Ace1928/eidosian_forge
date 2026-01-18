from unittest import mock
import fixtures
from oslotest import base as test_base
import webob.dec
import webob.exc
from oslo_middleware import catch_errors
def _test_has_request_id(self, application, expected_code=None):
    app = catch_errors.CatchErrors(application)
    req = webob.Request.blank('/test')
    req.environ['HTTP_X_AUTH_TOKEN'] = 'hello=world'
    res = req.get_response(app)
    self.assertEqual(expected_code, res.status_int)