import os
import re
import shutil
import subprocess
import stat
import string
import sys
def ExecManifestWrapper(self, arch, *args):
    """Run manifest tool with environment set. Strip out undesirable warning
    (some XML blocks are recognized by the OS loader, but not the manifest
    tool)."""
    env = self._GetEnv(arch)
    popen = subprocess.Popen(args, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = popen.communicate()[0].decode('utf-8')
    for line in out.splitlines():
        if line and 'manifest authoring warning 81010002' not in line:
            print(line)
    return popen.returncode