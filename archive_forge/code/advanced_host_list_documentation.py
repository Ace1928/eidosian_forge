from __future__ import (absolute_import, division, print_function)
import os
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.plugins.inventory import BaseInventoryPlugin
 parses the inventory file 