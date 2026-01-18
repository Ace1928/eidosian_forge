from unittest import mock
from oslo_serialization import jsonutils
import requests
import webob
from keystonemiddleware import ec2_token
from keystonemiddleware.tests.unit import utils
class EC2TokenMiddlewareTestGood(EC2TokenMiddlewareTestBase):

    @mock.patch.object(requests, 'post', return_value=FakeResponse(EMPTY_RESPONSE, status_code=200))
    def test_protocol_old_versions(self, mock_request):
        req = webob.Request.blank('/test')
        req.GET['Signature'] = 'test-signature'
        req.GET['AWSAccessKeyId'] = 'test-key-id'
        req.body = b'Action=ListUsers&Version=2010-05-08'
        resp = req.get_response(self.middleware)
        self.assertEqual(200, resp.status_code)
        self.assertEqual(TOKEN_ID, req.headers['X-Auth-Token'])
        mock_request.assert_called_with('http://localhost:5000/v3/ec2tokens', data=mock.ANY, headers={'Content-Type': 'application/json'}, verify=True, cert=None, timeout=mock.ANY)
        data = jsonutils.loads(mock_request.call_args[1]['data'])
        expected_data = {'ec2Credentials': {'access': 'test-key-id', 'headers': {'Host': 'localhost:80', 'Content-Length': '35'}, 'host': 'localhost:80', 'verb': 'GET', 'params': {'AWSAccessKeyId': 'test-key-id'}, 'signature': 'test-signature', 'path': '/test', 'body_hash': 'b6359072c78d70ebee1e81adcbab4f01bf2c23245fa365ef83fe8f1f955085e2'}}
        self.assertDictEqual(expected_data, data)

    @mock.patch.object(requests, 'post', return_value=FakeResponse(EMPTY_RESPONSE, status_code=200))
    def test_protocol_v4(self, mock_request):
        req = webob.Request.blank('/test')
        auth_str = 'AWS4-HMAC-SHA256 Credential=test-key-id/20110909/us-east-1/iam/aws4_request, SignedHeaders=content-type;host;x-amz-date, Signature=test-signature'
        req.headers['Authorization'] = auth_str
        req.body = b'Action=ListUsers&Version=2010-05-08'
        resp = req.get_response(self.middleware)
        self.assertEqual(200, resp.status_code)
        self.assertEqual(TOKEN_ID, req.headers['X-Auth-Token'])
        mock_request.assert_called_with('http://localhost:5000/v3/ec2tokens', data=mock.ANY, headers={'Content-Type': 'application/json'}, verify=True, cert=None, timeout=mock.ANY)
        data = jsonutils.loads(mock_request.call_args[1]['data'])
        expected_data = {'ec2Credentials': {'access': 'test-key-id', 'headers': {'Host': 'localhost:80', 'Content-Length': '35', 'Authorization': auth_str}, 'host': 'localhost:80', 'verb': 'GET', 'params': {}, 'signature': 'test-signature', 'path': '/test', 'body_hash': 'b6359072c78d70ebee1e81adcbab4f01bf2c23245fa365ef83fe8f1f955085e2'}}
        self.assertDictEqual(expected_data, data)