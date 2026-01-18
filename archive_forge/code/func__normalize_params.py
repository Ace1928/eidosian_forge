from __future__ import absolute_import, division, print_function
import copy
import json
from ansible.module_utils.six.moves.urllib import error as urllib_error
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.urls import open_url
def _normalize_params(params, arg_spec):
    final_params = {}
    for k, v in params.items():
        if k not in arg_spec:
            continue
        spec = arg_spec[k]
        if spec.get('type') == 'list' and spec.get('elements') == 'dict' and spec.get('options') and v:
            v = [_normalize_params(d, spec['options']) for d in v]
        elif spec.get('type') == 'dict' and spec.get('options') and v:
            v = _normalize_params(v, spec['options'])
        final_params[k] = v
    return final_params