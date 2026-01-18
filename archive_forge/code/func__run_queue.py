from __future__ import absolute_import, division, print_function
import collections
import json
import re
from ansible.module_utils._text import to_text
from ansible.module_utils.connection import ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
from ansible_collections.ansible.netcommon.plugins.plugin_utils.httpapi_base import HttpApiBase
def _run_queue(self, queue, output):
    if self._become:
        self.connection.queue_message('warning', 'become has no effect over httpapi. Use network_cli if privilege escalation is required')
    request = request_builder(queue, output)
    headers = {'Content-Type': 'application/json'}
    response, response_data = self.connection.send('/ins', request, headers=headers, method='POST')
    try:
        response_data = json.loads(to_text(response_data.getvalue()))
    except ValueError:
        raise ConnectionError('Response was not valid JSON, got {0}'.format(to_text(response_data.getvalue())))
    results = handle_response(response_data)
    return results