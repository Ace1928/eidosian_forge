from __future__ import division
import json
import os
import pickle
import collections
import contextlib
import warnings
import re
import io
import requests
import pytest
import urllib3
from requests.adapters import HTTPAdapter
from requests.auth import HTTPDigestAuth, _basic_auth_str
from requests.compat import (
from requests.cookies import (
from requests.exceptions import (
from requests.exceptions import SSLError as RequestsSSLError
from requests.models import PreparedRequest
from requests.structures import CaseInsensitiveDict
from requests.sessions import SessionRedirectMixin
from requests.models import urlencode
from requests.hooks import default_hooks
from requests.compat import JSONDecodeError, is_py3, MutableMapping
from .compat import StringIO, u
from .utils import override_environ
from urllib3.util import Timeout as Urllib3Timeout
def _patch_adapter_gzipped_redirect(self, session, url):
    adapter = session.get_adapter(url=url)
    org_build_response = adapter.build_response
    self._patched_response = False

    def build_response(*args, **kwargs):
        resp = org_build_response(*args, **kwargs)
        if not self._patched_response:
            resp.raw.headers['content-encoding'] = 'gzip'
            self._patched_response = True
        return resp
    adapter.build_response = build_response