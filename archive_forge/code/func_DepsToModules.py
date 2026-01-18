import os
import re
import subprocess
import gyp
import gyp.common
import gyp.xcode_emulation
from gyp.common import GetEnvironFallback
import hashlib
def DepsToModules(deps, prefix, suffix):
    modules = []
    for filepath in deps:
        filename = os.path.basename(filepath)
        if filename.startswith(prefix) and filename.endswith(suffix):
            modules.append(filename[len(prefix):-len(suffix)])
    return modules