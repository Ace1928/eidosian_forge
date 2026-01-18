import webob
from glance.api.middleware import context
import glance.context
from glance.tests.unit import base
class TestUnauthenticatedContextMiddleware(base.IsolatedUnitTest):

    def test_request(self):
        middleware = context.UnauthenticatedContextMiddleware(None)
        req = webob.Request.blank('/')
        middleware.process_request(req)
        self.assertIsNone(req.context.auth_token)
        self.assertIsNone(req.context.user_id)
        self.assertIsNone(req.context.project_id)
        self.assertEqual([], req.context.roles)
        self.assertTrue(req.context.is_admin)

    def test_response(self):
        middleware = context.UnauthenticatedContextMiddleware(None)
        req = webob.Request.blank('/')
        req.context = glance.context.RequestContext()
        request_id = req.context.request_id
        resp = webob.Response()
        resp.request = req
        middleware.process_response(resp)
        self.assertEqual(request_id, resp.headers['x-openstack-request-id'])
        resp_req_id = resp.headers['x-openstack-request-id']
        if isinstance(resp_req_id, bytes):
            resp_req_id = resp_req_id.decode('utf-8')
        self.assertFalse(resp_req_id.startswith('req-req-'))
        self.assertTrue(resp_req_id.startswith('req-'))