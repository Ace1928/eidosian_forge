import os
import sys
import hashlib
import filecmp
from collections import namedtuple
from shutil import which
from jedi.cache import memoize_method, time_cache
from jedi.inference.compiled.subprocess import CompiledSubprocess, \
import parso
def _get_subprocess(self):
    if self._subprocess is not None and (not self._subprocess.is_crashed):
        return self._subprocess
    try:
        self._subprocess = CompiledSubprocess(self._start_executable, env_vars=self._env_vars)
        info = self._subprocess._send(None, _get_info)
    except Exception as exc:
        raise InvalidPythonEnvironment('Could not get version information for %r: %r' % (self._start_executable, exc))
    self.executable = info[0]
    '\n        The Python executable, matches ``sys.executable``.\n        '
    self.path = info[1]
    '\n        The path to an environment, matches ``sys.prefix``.\n        '
    self.version_info = _VersionInfo(*info[2])
    "\n        Like :data:`sys.version_info`: a tuple to show the current\n        Environment's Python version.\n        "
    return self._subprocess