import ddt
from cinderclient import api_versions
from cinderclient import exceptions as exc
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
def _assert_call(self, base_url, detailed, params=None, method='GET', body=None):
    url = base_url
    if detailed:
        url += '/detail'
    if params:
        url += '?' + params
    if body:
        cs.assert_called(method, url, body)
    else:
        cs.assert_called(method, url)