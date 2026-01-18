from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
def _update_build_list(self):
    """creates self._build_list which is a unique list
        of build versions."""
    self._build_list = set()
    for node in self._nodes:
        build = self._info_cmd_helper('build', node)
        self._build_list.add(build)