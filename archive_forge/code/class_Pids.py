from __future__ import (absolute_import, division, print_function)
import abc
import re
from os.path import basename
from ansible.module_utils import six
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils import deps
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
class Pids(object):

    def __init__(self, module):
        deps.validate(module)
        self._ps = PSAdapter.from_package(psutil)
        self._module = module
        self._name = module.params['name']
        self._pattern = module.params['pattern']
        self._ignore_case = module.params['ignore_case']
        self._pids = []

    def execute(self):
        if self._name:
            self._pids = self._ps.get_pids_by_name(self._name)
        else:
            try:
                self._pids = self._ps.get_pids_by_pattern(self._pattern, self._ignore_case)
            except PSAdapterError as e:
                self._module.fail_json(msg=to_native(e))
        return self._module.exit_json(**self.result)

    @property
    def result(self):
        return {'pids': self._pids}