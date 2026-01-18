from __future__ import (absolute_import, division, print_function)
import json
import re
import sys
import datetime
import time
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six.moves.urllib.parse import urlencode
def _url_builder(self, path, params):
    d = self.module.params.get('query_parameters')
    if params is not None:
        d.update(params)
    query_string = urlencode(d, doseq=True)
    if path[0] == '/':
        path = path[1:]
    return '%s/%s?%s' % (self.module.params.get('api_url'), path, query_string)