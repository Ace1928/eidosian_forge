from __future__ import (absolute_import, division, print_function)
import json
import time
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.dns.plugins.module_utils.zone_record_api import (
def _process_json_result(self, content, info, must_have_content=True, method='GET', expected=None):
    if isinstance(must_have_content, (list, tuple)):
        must_have_content = info['status'] in must_have_content
    if info['status'] == 401:
        message = 'Unauthorized: the authentication parameters are incorrect (HTTP status 401)'
        try:
            body = json.loads(content.decode('utf8'))
            if body['message']:
                message = '{0}: {1}'.format(message, body['message'])
        except Exception:
            pass
        raise DNSAPIAuthenticationError(message)
    if info['status'] == 403:
        message = 'Forbidden: you do not have access to this resource (HTTP status 403)'
        try:
            body = json.loads(content.decode('utf8'))
            if body['message']:
                message = '{0}: {1}'.format(message, body['message'])
        except Exception:
            pass
        raise DNSAPIAuthenticationError(message)
    content_type = _get_header_value(info, 'content-type')
    if content_type != 'application/json' and (content_type is None or not content_type.startswith('application/json;')):
        if must_have_content:
            raise DNSAPIError('{0} {1} did not yield JSON data, but HTTP status code {2} with Content-Type "{3}" and data: {4}'.format(method, info['url'], info['status'], content_type, to_native(content)))
        self._validate(result=content, info=info, expected=expected, method=method)
        return (None, info)
    try:
        result = json.loads(content.decode('utf8'))
    except Exception:
        if must_have_content:
            raise DNSAPIError('{0} {1} did not yield JSON data, but HTTP status code {2} with data: {3}'.format(method, info['url'], info['status'], to_native(content)))
        self._validate(result=content, info=info, expected=expected, method=method)
        return (None, info)
    self._validate(result=result, info=info, expected=expected, method=method)
    return (result, info)