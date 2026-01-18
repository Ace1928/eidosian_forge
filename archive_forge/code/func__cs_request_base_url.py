from urllib import parse
from manilaclient import api_versions
from manilaclient.common import httpclient
from manilaclient.tests.unit import fakes
from manilaclient.tests.unit import utils
from manilaclient.v2 import client
def _cs_request_base_url(self, url, method, **kwargs):
    return self._cs_request_with_retries(url, method, **kwargs)