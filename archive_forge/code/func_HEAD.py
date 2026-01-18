from __future__ import absolute_import
import io
import json
import ssl
import certifi
import logging
import re
from six import PY3
from six.moves.urllib.parse import urlencode
def HEAD(self, url, headers=None, query_params=None, _preload_content=True, _request_timeout=None):
    return self.request('HEAD', url, headers=headers, _preload_content=_preload_content, _request_timeout=_request_timeout, query_params=query_params)