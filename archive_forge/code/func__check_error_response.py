import testtools
from unittest import mock
from troveclient.v1 import limits
def _check_error_response(self, status_code):
    RESPONSE_KEY = 'limits'
    resp = mock.Mock()
    resp.status_code = status_code
    body = {RESPONSE_KEY: {'absolute': {}, 'rate': [{'limit': []}]}}
    response = (resp, body)
    mock_get = mock.Mock(return_value=response)
    self.limits.api.client.get = mock_get
    self.assertRaises(Exception, self.limits.list)