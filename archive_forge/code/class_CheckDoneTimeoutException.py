from __future__ import absolute_import, division, print_function
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import PY3
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils.urls import fetch_url, open_url
import json
import time
class CheckDoneTimeoutException(Exception):

    def __init__(self, result, error):
        super(CheckDoneTimeoutException, self).__init__()
        self.result = result
        self.error = error