import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
def _dist_test_spawn_paths(self, cmd, display=None):
    """
        Fix msvc SDK ENV path same as distutils do
        without it we get c1: fatal error C1356: unable to find mspdbcore.dll
        """
    if not hasattr(self._ccompiler, '_paths'):
        self._dist_test_spawn(cmd)
        return
    old_path = os.getenv('path')
    try:
        os.environ['path'] = self._ccompiler._paths
        self._dist_test_spawn(cmd)
    finally:
        os.environ['path'] = old_path