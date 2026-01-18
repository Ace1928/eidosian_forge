from troveclient import base
from troveclient import common
from troveclient.v1 import users
def _disable_root(self, url):
    resp, body = self.api.client.delete(url)
    common.check_for_exceptions(resp, body, url)