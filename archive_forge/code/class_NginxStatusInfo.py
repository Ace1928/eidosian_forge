from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.text.converters import to_text
class NginxStatusInfo(object):

    def __init__(self):
        self.url = module.params.get('url')
        self.timeout = module.params.get('timeout')

    def run(self):
        result = {'active_connections': None, 'accepts': None, 'handled': None, 'requests': None, 'reading': None, 'writing': None, 'waiting': None, 'data': None}
        response, info = fetch_url(module=module, url=self.url, force=True, timeout=self.timeout)
        if not response:
            module.fail_json(msg='No valid or no response from url %s within %s seconds (timeout)' % (self.url, self.timeout))
        data = to_text(response.read(), errors='surrogate_or_strict')
        if not data:
            return result
        result['data'] = data
        expr = 'Active connections: ([0-9]+) \\nserver accepts handled requests\\n ([0-9]+) ([0-9]+) ([0-9]+) \\nReading: ([0-9]+) Writing: ([0-9]+) Waiting: ([0-9]+)'
        match = re.match(expr, data, re.S)
        if match:
            result['active_connections'] = int(match.group(1))
            result['accepts'] = int(match.group(2))
            result['handled'] = int(match.group(3))
            result['requests'] = int(match.group(4))
            result['reading'] = int(match.group(5))
            result['writing'] = int(match.group(6))
            result['waiting'] = int(match.group(7))
        return result