from __future__ import absolute_import, division, print_function
import copy
import glob
import os
from importlib import import_module
from ansible.errors import AnsibleActionFail, AnsibleError
from ansible.module_utils._text import to_text
from ansible.utils.display import Display
from ansible_collections.ansible.netcommon.plugins.action.network import (
def _get_resource_module(self, prefix_os_name=False):
    if '.' in self._name:
        if len(self._name.split('.')) != 3:
            msg = 'name should a fully qualified collection name in the format <org-name>.<collection-name>.<resource-module-name>'
            raise AnsibleActionFail(msg)
        fqcn_module_name = self._name
    else:
        if prefix_os_name:
            module_name = self._os_name.split('.')[1] + '_' + self._name
        else:
            module_name = self._name
        fqcn_module_name = '.'.join(self._os_name.split('.')[:2] + [module_name])
    return fqcn_module_name