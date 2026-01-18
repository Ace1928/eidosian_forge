import os
import re
import shutil
import subprocess
import stat
import string
import sys
def ExecLinkWrapper(self, arch, use_separate_mspdbsrv, *args):
    """Filter diagnostic output from link that looks like:
    '   Creating library ui.dll.lib and object ui.dll.exp'
    This happens when there are exports from the dll or exe.
    """
    env = self._GetEnv(arch)
    if use_separate_mspdbsrv == 'True':
        self._UseSeparateMspdbsrv(env, args)
    if sys.platform == 'win32':
        args = list(args)
        args[0] = args[0].replace('/', '\\')
    link = subprocess.Popen(args, shell=sys.platform == 'win32', env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = link.communicate()[0].decode('utf-8')
    for line in out.splitlines():
        if not line.startswith('   Creating library ') and (not line.startswith('Generating code')) and (not line.startswith('Finished generating code')):
            print(line)
    return link.returncode