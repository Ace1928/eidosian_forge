from __future__ import (absolute_import, division, print_function)
import json
from ansible.plugins.httpapi import HttpApiBase
from ansible.module_utils.basic import to_text
from ansible.module_utils.six.moves import urllib
import re
from datetime import datetime
def _concat_params(self, url, params):
    if not params or not len(params):
        return url
    url = url + '?' if '?' not in url else url
    for param_key in params:
        param_value = params[param_key]
        if url[-1] == '?':
            url += '%s=%s' % (param_key, param_value)
        else:
            url += '&%s=%s' % (param_key, param_value)
    return url