from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.general.plugins.module_utils.module_helper import StateModuleHelper
from ansible_collections.community.general.plugins.module_utils.snap import snap_runner
def _has_alias(self, name=None, alias=None):
    if name:
        if name not in self.vars.snap_aliases:
            return False
        if alias is None:
            return bool(self.vars.snap_aliases[name])
        return alias in self.vars.snap_aliases[name]
    return any((alias in aliases for aliases in self.vars.snap_aliases.values()))