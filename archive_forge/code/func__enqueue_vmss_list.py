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
def _enqueue_vmss_list(self, rg=None):
    if not rg or rg == '*':
        url = '/subscriptions/{subscriptionId}/providers/Microsoft.Compute/virtualMachineScaleSets'
    else:
        url = '/subscriptions/{subscriptionId}/resourceGroups/{rg}/providers/Microsoft.Compute/virtualMachineScaleSets'
    url = url.format(subscriptionId=self._clientconfig.subscription_id, rg=rg)
    self._enqueue_get(url=url, api_version=self._compute_api_version, handler=self._on_vmss_page_response)