from __future__ import absolute_import, division, print_function
import json
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import missing_required_lib
from ..module_utils.gcp_utils import (
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
def _return_if_object(self, module, response):
    """
        :param module: A GcpModule
        :param response: A Requests response object
        :return JSON response
        """
    if response.status_code == 404:
        return None
    if response.status_code == 204:
        return None
    try:
        response.raise_for_status
        result = response.json()
    except getattr(json.decoder, 'JSONDecodeError', ValueError) as inst:
        module.fail_json(msg='Invalid JSON response with error: %s' % inst)
    except GcpRequestException as inst:
        module.fail_json(msg='Network error: %s' % inst)
    if navigate_hash(result, ['error', 'errors']):
        module.fail_json(msg=navigate_hash(result, ['error', 'errors']))
    return result