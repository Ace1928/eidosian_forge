from __future__ import absolute_import, division, print_function
import ast
import base64
import json
import os
import re
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.connection import ConnectionError
from ansible.plugins.httpapi import HttpApiBase
from copy import copy, deepcopy
def _return_info(self, response_code, method, path, msg, respond_data=None, error=None):
    """Format success/error data and return with consistent format"""
    info = {}
    info['status'] = response_code
    info['method'] = method
    info['url'] = path
    info['msg'] = msg
    if error is not None:
        info['error'] = error
    else:
        info['error'] = {}
    if respond_data is not None:
        info['body'] = respond_data
    return info