from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
def _lookup_entry(self, module, request_url):
    """
        Lookup for existing entry
        :param module:
        :param request_url:
        :return:
        """
    response, info = fetch_url(module, request_url, method='GET', headers={'Accept': 'application/json'})
    result = None
    if response is not None:
        results = json.loads(response.read())
        result = next(iter(filter(lambda d: d['name'] == module.params.get('name'), results)), None)
    return (info, result)