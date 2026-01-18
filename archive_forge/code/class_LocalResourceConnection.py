from __future__ import absolute_import, division, print_function
import json
import traceback
from ansible.module_utils._text import to_native, to_text
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils.connection import Connection, ConnectionError
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.netconf import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.parsing import Cli
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_list
class LocalResourceConnection:

    def __init__(self, module):
        self.module = module

    def get(self, *args, **kwargs):
        self.module.fail_json(msg='Network resource modules not supported over local connection.')