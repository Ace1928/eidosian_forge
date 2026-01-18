import uuid
from oslotest import base as test_base
from testtools import matchers
import webob
import webob.dec
from oslo_middleware import request_id
class RequestIdTest(test_base.BaseTestCase):

    def test_generate_request_id(self):

        @webob.dec.wsgify
        def application(req):
            return req.environ[request_id.ENV_REQUEST_ID]
        app = request_id.RequestId(application)
        req = webob.Request.blank('/test')
        res = req.get_response(app)
        res_req_id = res.headers.get(request_id.HTTP_RESP_HEADER_REQUEST_ID)
        if isinstance(res_req_id, bytes):
            res_req_id = res_req_id.decode('utf-8')
        self.assertThat(res_req_id, matchers.StartsWith('req-'))
        self.assertEqual(res.body.decode('utf-8'), res_req_id)

    def test_compat_headers(self):
        """Test that compat headers are set

        Compat headers might exist on a super class to support
        previous API contracts. This ensures that you can set that to
        a list of headers and those values are the same as the
        request_id.

        """

        @webob.dec.wsgify
        def application(req):
            return req.environ[request_id.ENV_REQUEST_ID]
        app = AltHeader(application)
        req = webob.Request.blank('/test')
        res = req.get_response(app)
        res_req_id = res.headers.get(request_id.HTTP_RESP_HEADER_REQUEST_ID)
        self.assertEqual(res.headers.get('x-compute-req-id'), res_req_id)
        self.assertEqual(res.headers.get('x-silly-id'), res_req_id)

    def test_global_request_id_set(self):
        """Test that global request_id is set."""

        @webob.dec.wsgify
        def application(req):
            return req.environ[request_id.GLOBAL_REQ_ID]
        global_req = 'req-%s' % uuid.uuid4()
        app = request_id.RequestId(application)
        req = webob.Request.blank('/test', headers={'X-OpenStack-Request-ID': global_req})
        res = req.get_response(app)
        res_req_id = res.headers.get(request_id.HTTP_RESP_HEADER_REQUEST_ID)
        if isinstance(res_req_id, bytes):
            res_req_id = res_req_id.decode('utf-8')
        self.assertEqual(res.body.decode('utf-8'), global_req)
        self.assertNotEqual(res.body.decode('utf-8'), res_req_id)

    def test_global_request_id_drop(self):
        """Test that bad format ids are dropped.

        This ensures that badly formatted ids are dropped entirely.
        """

        @webob.dec.wsgify
        def application(req):
            return req.environ.get(request_id.GLOBAL_REQ_ID)
        global_req = 'req-%s-bad' % uuid.uuid4()
        app = request_id.RequestId(application)
        req = webob.Request.blank('/test', headers={'X-OpenStack-Request-ID': global_req})
        res = req.get_response(app)
        res_req_id = res.headers.get(request_id.HTTP_RESP_HEADER_REQUEST_ID)
        if isinstance(res_req_id, bytes):
            res_req_id = res_req_id.decode('utf-8')
        self.assertEqual(res.body.decode('utf-8'), '')