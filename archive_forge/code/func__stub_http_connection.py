import json
from unittest import mock
from oslo_config import cfg
from oslo_utils import importutils
import requests
from heat.api.aws import ec2token
from heat.api.aws import exception
from heat.common import wsgi
from heat.tests import common
from heat.tests import utils
def _stub_http_connection(self, headers=None, params=None, response=None, req_url='http://123:5000/v3/ec2tokens', verify=True, cert=None, direct_mock=True):
    headers = headers or {}
    params = params or {}

    class DummyHTTPResponse(object):
        text = response
        headers = {'X-Subject-Token': 123}

        def json(self):
            return json.loads(self.text)
    body_hash = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    req_creds = {'ec2Credentials': {'access': 'foo', 'headers': headers, 'host': 'heat:8000', 'verb': 'GET', 'params': params, 'signature': 'xyz', 'path': '/v1', 'body_hash': body_hash}}
    req_headers = {'Content-Type': 'application/json'}
    self.verify_req_url = req_url
    self.verify_data = utils.JsonRepr(req_creds)
    self.verify_verify = verify
    self.verify_cert = cert
    self.verify_req_headers = req_headers
    if direct_mock:
        requests.post.return_value = DummyHTTPResponse()
    else:
        return DummyHTTPResponse()