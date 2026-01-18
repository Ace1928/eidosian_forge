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
def _on_vm_page_response(self, response, vmss=None):
    next_link = response.get('nextLink')
    if next_link:
        self._enqueue_get(url=next_link, api_version=self._compute_api_version, handler=self._on_vm_page_response)
    if 'value' in response:
        for h in response['value']:
            self._hosts.append(AzureHost(h, self, vmss=vmss, legacy_name=self._legacy_hostnames))