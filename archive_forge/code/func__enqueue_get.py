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
def _enqueue_get(self, url, api_version, handler, handler_args=None):
    if not handler_args:
        handler_args = {}
    self._request_queue.put_nowait(UrlAction(url=url, api_version=api_version, handler=handler, handler_args=handler_args))