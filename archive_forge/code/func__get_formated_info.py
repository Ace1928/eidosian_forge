from __future__ import absolute_import, division, print_function
import json
import re
import traceback
from ansible.module_utils.six import PY3
from ansible.module_utils._text import to_text
from ansible.module_utils.connection import ConnectionError
from ansible.plugins.httpapi import HttpApiBase
from copy import copy
def _get_formated_info(self, response):
    """The code in this function is based out of Ansible fetch_url code
        at https://github.com/ansible/ansible/blob/devel/lib/ansible/module_utils/urls.py"""
    info = dict(msg='OK (%s bytes)' % response.headers.get('Content-Length', 'unknown'), url=response.geturl(), status=response.getcode())
    info.update(dict(((k.lower(), v) for k, v in response.info().items())))
    if PY3:
        temp_headers = {}
        for name, value in response.headers.items():
            name = name.lower()
            if name in temp_headers:
                temp_headers[name] = ', '.join((temp_headers[name], value))
            else:
                temp_headers[name] = value
        info.update(temp_headers)
    return info