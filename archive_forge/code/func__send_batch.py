from __future__ import (absolute_import, division, print_function)
import hashlib
import json
import re
import uuid
import os
from collections import namedtuple
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable
from ansible.module_utils.six import iteritems
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMAuth
from ansible.errors import AnsibleParserError, AnsibleError
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils._text import to_native, to_bytes, to_text
from itertools import chain
def _send_batch(self, batched_requests):
    url = '/batch'
    query_parameters = {'api-version': '2015-11-01'}
    header_parameters = {'x-ms-client-request-id': str(uuid.uuid4()), 'Content-Type': 'application/json; charset=utf-8'}
    body_content = dict(requests=batched_requests)
    header = {'x-ms-client-request-id': str(uuid.uuid4())}
    header.update(self._default_header_parameters)
    request_new = self.new_client.post(url, query_parameters, header_parameters, body_content)
    response = self.new_client.send_request(request_new)
    return json.loads(response.body())