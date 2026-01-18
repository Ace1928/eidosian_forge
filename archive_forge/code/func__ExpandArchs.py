import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def _ExpandArchs(self, archs, sdkroot):
    """Expands variables references in ARCHS, and remove duplicates."""
    variable_mapping = self._VariableMapping(sdkroot)
    expanded_archs = []
    for arch in archs:
        if self.variable_pattern.match(arch):
            variable = arch
            try:
                variable_expansion = variable_mapping[variable]
                for arch in variable_expansion:
                    if arch not in expanded_archs:
                        expanded_archs.append(arch)
            except KeyError:
                print('Warning: Ignoring unsupported variable "%s".' % variable)
        elif arch not in expanded_archs:
            expanded_archs.append(arch)
    return expanded_archs