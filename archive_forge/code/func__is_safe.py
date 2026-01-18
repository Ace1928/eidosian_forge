import os
import sys
import hashlib
import filecmp
from collections import namedtuple
from shutil import which
from jedi.cache import memoize_method, time_cache
from jedi.inference.compiled.subprocess import CompiledSubprocess, \
import parso
def _is_safe(executable_path):
    real_path = os.path.realpath(executable_path)
    if _is_unix_safe_simple(real_path):
        return True
    for environment in find_system_environments():
        if environment.executable == real_path:
            return True
        if environment._sha256 == _calculate_sha256_for_file(real_path):
            return True
    return False