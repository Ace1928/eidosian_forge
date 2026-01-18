import io
import logging
import os.path
import urllib.parse
from smart_open import bytebuffer, constants
import smart_open.utils
def _partial_request(self, start_pos=None):
    if start_pos is not None:
        self.headers.update({'range': smart_open.utils.make_range_string(start_pos)})
    response = requests.get(self.url, auth=self.auth, stream=True, cert=self.cert, headers=self.headers, timeout=self.timeout)
    return response