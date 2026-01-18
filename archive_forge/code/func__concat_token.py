from __future__ import (absolute_import, division, print_function)
import json
from ansible.plugins.httpapi import HttpApiBase
from ansible.module_utils.basic import to_text
from ansible.module_utils.six.moves import urllib
import re
from datetime import datetime
def _concat_token(self, url):
    if self.get_access_token():
        token_pair = 'access_token=' + self.get_access_token()
        return url + '&' + token_pair if '?' in url else url + '?' + token_pair
    return url