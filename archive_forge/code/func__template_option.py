from __future__ import absolute_import, division, print_function
import json
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.inventory.group import to_safe_group_name
from ansible.module_utils._text import to_native
from ansible.module_utils.urls import Request
from ansible.module_utils.six.moves.urllib.error import URLError, HTTPError
from ansible.plugins.inventory import BaseInventoryPlugin, Constructable, Cacheable
def _template_option(self, option):
    value = self.get_option(option)
    self.templar.available_variables = {}
    return self.templar.template(value)