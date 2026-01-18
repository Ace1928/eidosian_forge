import os
import re
import shutil
import subprocess
import stat
import string
import sys
def ExecActionWrapper(self, arch, rspfile, *dir):
    """Runs an action command line from a response file using the environment
    for |arch|. If |dir| is supplied, use that as the working directory."""
    env = self._GetEnv(arch)
    for k, v in os.environ.items():
        if k not in env:
            env[k] = v
    args = open(rspfile).read()
    dir = dir[0] if dir else None
    return subprocess.call(args, shell=True, env=env, cwd=dir)