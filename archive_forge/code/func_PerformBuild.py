import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def PerformBuild(data, configurations, params):
    options = params['options']
    for config in configurations:
        arguments = ['make']
        if options.toplevel_dir and options.toplevel_dir != '.':
            arguments += ('-C', options.toplevel_dir)
        arguments.append('BUILDTYPE=' + config)
        print(f'Building [{config}]: {arguments}')
        subprocess.check_call(arguments)