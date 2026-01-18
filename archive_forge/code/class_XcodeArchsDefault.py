import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
class XcodeArchsDefault:
    """A class to resolve ARCHS variable from xcode_settings, resolving Xcode
  macros and implementing filtering by VALID_ARCHS. The expansion of macros
  depends on the SDKROOT used ("macosx", "iphoneos", "iphonesimulator") and
  on the version of Xcode.
  """
    variable_pattern = re.compile('\\$\\([a-zA-Z_][a-zA-Z0-9_]*\\)$')

    def __init__(self, default, mac, iphonesimulator, iphoneos):
        self._default = (default,)
        self._archs = {'mac': mac, 'ios': iphoneos, 'iossim': iphonesimulator}

    def _VariableMapping(self, sdkroot):
        """Returns the dictionary of variable mapping depending on the SDKROOT."""
        sdkroot = sdkroot.lower()
        if 'iphoneos' in sdkroot:
            return self._archs['ios']
        elif 'iphonesimulator' in sdkroot:
            return self._archs['iossim']
        else:
            return self._archs['mac']

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

    def ActiveArchs(self, archs, valid_archs, sdkroot):
        """Expands variables references in ARCHS, and filter by VALID_ARCHS if it
    is defined (if not set, Xcode accept any value in ARCHS, otherwise, only
    values present in VALID_ARCHS are kept)."""
        expanded_archs = self._ExpandArchs(archs or self._default, sdkroot or '')
        if valid_archs:
            filtered_archs = []
            for arch in expanded_archs:
                if arch in valid_archs:
                    filtered_archs.append(arch)
            expanded_archs = filtered_archs
        return expanded_archs