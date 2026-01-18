import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def ExtractSharedMSVSSystemIncludes(configs, generator_flags):
    """Finds msvs_system_include_dirs that are common to all targets, removes
    them from all targets, and returns an OrderedSet containing them."""
    all_system_includes = OrderedSet(configs[0].get('msvs_system_include_dirs', []))
    for config in configs[1:]:
        system_includes = config.get('msvs_system_include_dirs', [])
        all_system_includes = all_system_includes & OrderedSet(system_includes)
    if not all_system_includes:
        return None
    env = GetGlobalVSMacroEnv(GetVSVersion(generator_flags))
    expanded_system_includes = OrderedSet([ExpandMacros(include, env) for include in all_system_includes])
    if any(['$' in include for include in expanded_system_includes]):
        return None
    for config in configs:
        includes = config.get('msvs_system_include_dirs', [])
        if includes:
            new_includes = [i for i in includes if i not in all_system_includes]
            config['msvs_system_include_dirs'] = new_includes
    return expanded_system_includes