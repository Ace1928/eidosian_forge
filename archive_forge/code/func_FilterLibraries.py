import gyp
import gyp.common
import gyp.generator.make as make  # Reuse global functions from make backend.
import os
import re
import subprocess
def FilterLibraries(self, libraries):
    """Filter the 'libraries' key to separate things that shouldn't be ldflags.

        Library entries that look like filenames should be converted to android
        module names instead of being passed to the linker as flags.

        Args:
          libraries: the value of spec.get('libraries')
        Returns:
          A tuple (static_lib_modules, dynamic_lib_modules, ldflags)
        """
    static_lib_modules = []
    dynamic_lib_modules = []
    ldflags = []
    for libs in libraries:
        for lib in libs.split():
            if lib == '-lc' or lib == '-lstdc++' or lib == '-lm' or lib.endswith('libgcc.a'):
                continue
            match = re.search('([^/]+)\\.a$', lib)
            if match:
                static_lib_modules.append(match.group(1))
                continue
            match = re.search('([^/]+)\\.so$', lib)
            if match:
                dynamic_lib_modules.append(match.group(1))
                continue
            if lib.startswith('-l'):
                ldflags.append(lib)
    return (static_lib_modules, dynamic_lib_modules, ldflags)