from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.general.plugins.module_utils.module_helper import StateModuleHelper
from ansible_collections.community.general.plugins.module_utils.snap import snap_runner
def _get_aliases_for(self, name):
    return self._get_aliases().get(name, [])