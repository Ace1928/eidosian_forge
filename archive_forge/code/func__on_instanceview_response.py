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
def _on_instanceview_response(self, vm_instanceview_model):
    self._instanceview = vm_instanceview_model
    self._powerstate = next((self._powerstate_regex.match(s.get('code', '')).group('powerstate') for s in vm_instanceview_model.get('statuses', []) if self._powerstate_regex.match(s.get('code', ''))), 'unknown')