from __future__ import (absolute_import, division, print_function)
import os
import time
import traceback
from ansible.module_utils._text import to_text
import json
from ansible_collections.fortinet.fortios.plugins.module_utils.common.type_utils import underscore_to_hyphen
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.secret_field import is_secret_field
def _validate_member_parameter(self, trace, trace_param, trace_url_tokens, attr_blobs, attr_params):
    attr_blob = attr_blobs[0]
    current_attr_name = attr_blob['name']
    current_attr_mkey = attr_blob['mkey']
    trace.append(current_attr_name)
    if not attr_params:
        self._module.fail_json('parameter %s is empty' % self._trace_to_string(trace))
    if type(attr_params) not in [list, dict]:
        raise AssertionError('Invalid attribute type')
    if type(attr_params) is dict:
        trace_param_item = dict()
        trace_param_item[current_attr_name] = (None, attr_params)
        trace_param.append(trace_param_item)
        if len(attr_blobs) <= 1:
            raise AssertionError('Invalid attribute blob')
        next_attr_blob = attr_blobs[1]
        next_attr_name = next_attr_blob['name']
        self._validate_member_parameter(trace, trace_param, trace_url_tokens, attr_blobs[1:], attr_params[next_attr_name])
        del trace_param[-1]
        return
    for param in attr_params:
        if current_attr_mkey not in param or not param[current_attr_mkey]:
            self._module.fail_json('parameter %s.%s is empty' % (self._trace_to_string(trace), current_attr_mkey))
        trace_param_item = dict()
        trace_param_item[current_attr_name] = (param[current_attr_mkey], param)
        trace_param.append(trace_param_item)
        if len(attr_blobs) > 1:
            next_attr_blob = attr_blobs[1]
            next_attr_name = next_attr_blob['name']
            if next_attr_name in param:
                self._validate_member_parameter(trace, trace_param, trace_url_tokens, attr_blobs[1:], param[next_attr_name])
            else:
                url_tokens = list()
                for token in trace_param:
                    url_tokens.append(token)
                trace_url_tokens.append(url_tokens)
        else:
            url_tokens = list()
            for token in trace_param:
                url_tokens.append(token)
            trace_url_tokens.append(url_tokens)
        del trace_param[-1]