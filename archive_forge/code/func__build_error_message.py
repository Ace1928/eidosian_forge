from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.urls import fetch_url
def _build_error_message(self, module, info):
    s = ''
    body = info.get('body')
    if body:
        errors = module.from_json(body).get('errors')
        if errors:
            error = errors[0]
            name = error.get('name')
            if name:
                s += '{0} :'.format(name)
            description = error.get('description')
            if description:
                s += description
    return s