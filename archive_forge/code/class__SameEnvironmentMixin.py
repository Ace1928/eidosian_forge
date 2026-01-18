import os
import sys
import hashlib
import filecmp
from collections import namedtuple
from shutil import which
from jedi.cache import memoize_method, time_cache
from jedi.inference.compiled.subprocess import CompiledSubprocess, \
import parso
class _SameEnvironmentMixin:

    def __init__(self):
        self._start_executable = self.executable = sys.executable
        self.path = sys.prefix
        self.version_info = _VersionInfo(*sys.version_info[:3])
        self._env_vars = None