from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
class UTMModuleConfigurationError(Exception):

    def __init__(self, msg, **args):
        super(UTMModuleConfigurationError, self).__init__(self, msg)
        self.msg = msg
        self.module_fail_args = args

    def do_fail(self, module):
        module.fail_json(msg=self.msg, other=self.module_fail_args)