import testtools
from unittest import mock
from troveclient.v1 import databases
def _get_mock_method(self):
    self._resp = mock.Mock()
    self._body = None
    self._url = None

    def side_effect_func(url, body=None):
        self._body = body
        self._url = url
        return (self._resp, body)
    return mock.Mock(side_effect=side_effect_func)